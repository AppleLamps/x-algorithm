# X's For You Feed Algorithm

This document explains how X's recommendation algorithm works to build your personalized For You timeline.

## Overview

The For You feed combines posts from two sources:

1. **In-Network (Thunder)**: Posts from accounts you follow
2. **Out-of-Network (Phoenix)**: Posts discovered via machine learning from the global corpus

These candidates are then scored, filtered, and ranked to produce your personalized feed.

---

## High-Level Pipeline

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────┐     ┌────────┐
│   Thunder   │────▶│              │     │              │     │          │     │        │
│ (In-Network)│     │    Hydrate   │────▶│    Filter    │────▶│   Score  │────▶│  Rank  │──▶ Feed
│             │     │   Candidates │     │  Candidates  │     │          │     │        │
│   Phoenix   │────▶│              │     │              │     │          │     │        │
│(Out-Network)│     └──────────────┘     └──────────────┘     └──────────┘     └────────┘
└─────────────┘
```

---

## Stage 1: Candidate Retrieval

### Thunder (In-Network)

Thunder retrieves recent posts from accounts you follow:

- Stores posts in-memory for sub-millisecond retrieval
- Ingests posts in real-time from Kafka
- Returns posts sorted by recency
- Includes original posts and replies

### Phoenix Retrieval (Out-of-Network)

Phoenix uses a **two-tower neural network** to find relevant posts from millions of candidates:

```
     User Tower                         Candidate Tower
         │                                    │
         ▼                                    ▼
  ┌─────────────────┐                ┌─────────────────┐
  │  User Features  │                │  Post Features  │
  │  + Engagement   │                │  + Author Info  │
  │     History     │                │                 │
  └────────┬────────┘                └────────┬────────┘
           │                                  │
           ▼                                  ▼
    Transformer                          MLP Tower
    Encoder                              (2-layer)
           │                                  │
           ▼                                  ▼
  [Normalized User                   [Normalized Post
   Embedding]                         Embeddings]
           │                                  │
           └──────────┬───────────────────────┘
                      ▼
              Dot Product Similarity
                      │
                      ▼
               Top-K Candidates
```

The user tower processes:

- User ID (hashed for embedding lookup)
- Recent engagement history (posts viewed, liked, replied to)
- Actions taken (like, repost, reply, etc.)
- Product surface context (where the action occurred)

---

## Stage 2: Candidate Hydration

After retrieval, candidates are enriched with additional metadata:

| Hydrator | Purpose |
|----------|---------|
| Core Data | Fetch tweet text, metadata from TweetyPie |
| In-Network | Mark whether author is followed |
| Author Info | Get author followers count, screen name |
| Video Duration | Fetch video length for video content |
| Subscription | Check if author has subscriptions |

---

## Stage 3: Filtering

Candidates are filtered to remove ineligible content:

| Filter | Purpose |
|--------|---------|
| Age Filter | Remove posts older than threshold |
| Drop Duplicates | Remove duplicate tweet IDs |
| Self Tweet | Exclude user's own posts |
| Previously Seen | Exclude posts user has already seen |
| Previously Served | Exclude recently served posts |
| Author Social Graph | Block/mute relationship checks |
| Muted Keywords | Remove posts with muted words |
| Retweet Deduplication | Remove duplicate retweets |
| Conversation Dedup | Limit posts per conversation |
| Visibility Filter | Safety and policy enforcement |

---

## Stage 4: Scoring with Phoenix Ranking

The **Phoenix Ranking Model** predicts engagement probabilities using a Grok-based transformer architecture.

### Input Construction

The model receives three components concatenated as a sequence:

```
[User Embedding] + [History Embeddings] + [Candidate Embeddings]
      [1, D]            [S, D]                 [C, D]
```

Each embedding combines:

- **Post hashes** → multiple hash functions for robust embedding lookup
- **Author hashes** → author identity embedding
- **Product surface** → where the content was seen/created
- **Actions** (history only) → what engagement occurred

### Candidate Isolation via Attention Masking

A critical design feature: **candidates cannot attend to each other**. This ensures that a post's score doesn't depend on which other posts are in the batch.

```
         Keys (what we attend TO)
         ───────────────────────────────────────────────▶
         │ User │    History (S)    │   Candidates (C)   │
    ┌────┼──────┼───────────────────┼────────────────────┤
 Q  │ U  │  ✓   │  ✓   ✓   ✓   ✓   │  ✗   ✗   ✗   ✗    │
 u  ├────┼──────┼───────────────────┼────────────────────┤
 e  │ H  │  ✓   │  ✓   ✓   ✓   ✓   │  ✗   ✗   ✗   ✗    │
 r  │    │  ✓   │  ✓   ✓   ✓   ✓   │  ✗   ✗   ✗   ✗    │
 i  ├────┼──────┼───────────────────┼────────────────────┤
 e  │ C  │  ✓   │  ✓   ✓   ✓   ✓   │  ✓   ✗   ✗   ✗    │
 s  │    │  ✓   │  ✓   ✓   ✓   ✓   │  ✗   ✓   ✗   ✗    │
 │  │    │  ✓   │  ✓   ✓   ✓   ✓   │  ✗   ✗   ✓   ✗    │
 ▼  │    │  ✓   │  ✓   ✓   ✓   ✓   │  ✗   ✗   ✗   ✓    │
    └────┴──────┴───────────────────┴────────────────────┘

    ✓ = Can attend     ✗ = Cannot attend
```

- User + History: Full bidirectional attention
- Candidates → User/History: Candidates CAN see context
- Candidates → Candidates: Only self-attention (diagonal)

### Multi-Action Prediction

The model predicts probabilities for multiple engagement types simultaneously:

```python
Output: [batch_size, num_candidates, num_actions]

Actions predicted:
├── favorite_score      # Probability of liking
├── reply_score         # Probability of replying
├── retweet_score       # Probability of reposting
├── click_score         # Probability of clicking
├── profile_click_score # Probability of viewing author profile
├── vqv_score           # Video quality view (watched >50%)
├── share_score         # Probability of sharing
├── dwell_score         # Probability of extended viewing
├── quote_score         # Probability of quote tweeting
├── follow_author_score # Probability of following author
└── Negative signals:
    ├── not_interested_score
    ├── block_author_score
    ├── mute_author_score
    └── report_score
```

---

## Stage 5: Weighted Score Computation

Individual engagement predictions are combined into a single score:

```rust
weighted_score =
    favorite_score      × FAVORITE_WEIGHT
  + reply_score         × REPLY_WEIGHT
  + retweet_score       × RETWEET_WEIGHT
  + click_score         × CLICK_WEIGHT
  + profile_click_score × PROFILE_CLICK_WEIGHT
  + vqv_score           × VQV_WEIGHT           // if video > min duration
  + share_score         × SHARE_WEIGHT
  + dwell_score         × DWELL_WEIGHT
  + quote_score         × QUOTE_WEIGHT
  + follow_author_score × FOLLOW_AUTHOR_WEIGHT
  // Negative actions subtract from score
  - not_interested_score × NOT_INTERESTED_WEIGHT
  - block_author_score   × BLOCK_AUTHOR_WEIGHT
  - mute_author_score    × MUTE_AUTHOR_WEIGHT
  - report_score         × REPORT_WEIGHT
```

---

## Stage 6: Diversity & Network Adjustments

### Author Diversity Scorer

Prevents feed domination by single authors using exponential decay:

```
multiplier = (1 - floor) × decay^position + floor
```

For each additional post by the same author, their score is multiplied by an increasingly smaller factor.

### Out-of-Network Weight

In-network posts (from followed accounts) receive priority over out-of-network posts via a weight factor applied to OON candidates.

---

## Stage 7: Selection & Final Ranking

1. **Sort** candidates by final score (descending)
2. **Select** top K candidates
3. **Post-selection filters** apply final visibility checks
4. **Side effects** cache request info for future use

---

## Key Design Principles

### 1. No Hand-Engineered Features

The Grok-based transformer learns relevance directly from user engagement sequences. No manual feature engineering.

### 2. Hash-Based Embeddings

Uses multiple hash functions per entity for:

- Efficient lookup
- Collision resistance
- Memory efficiency

### 3. Real-Time Adaptation

- Thunder ingests posts in real-time
- User action sequences capture recent behavior
- Scoring happens on every request

### 4. Balanced Objectives

Weights can be adjusted to balance:

- Engagement optimization
- User satisfaction (negative signal weights)
- Content diversity

---

## Architecture Components

| Component | Language | Purpose |
|-----------|----------|---------|
| Home Mixer | Rust | Main orchestration service |
| Phoenix | Python/JAX | ML ranking & retrieval models |
| Thunder | Rust | In-network post storage & retrieval |
| Candidate Pipeline | Rust | Reusable pipeline framework |

---

## Request Flow Summary

1. **User opens For You tab** → gRPC request to Home Mixer
2. **Parallel fetch** from Thunder (followed accounts) and Phoenix (ML retrieval)
3. **Hydrate** candidates with metadata
4. **Filter** ineligible content
5. **Score** with Phoenix transformer model
6. **Weight** engagement predictions
7. **Diversify** by author
8. **Rank** and select top posts
9. **Return** personalized feed
