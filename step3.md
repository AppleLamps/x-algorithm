# Step 3: Filtering Out Unwanted Posts

This document provides a detailed explanation of how X's recommendation algorithm filters posts before they appear in your For You feed. Filtering ensures you only see relevant, high-quality content that respects your preferences and safety standards.

---

## Overview

Filtering occurs at **two stages** in the pipeline:

1. **Pre-Scoring Filters**: Run before ML scoring to reduce candidate pool
2. **Post-Selection Filters**: Run after scoring and selection for final safety checks

```
                         FILTERING PIPELINE
┌──────────────────────────────────────────────────────────────────────┐
│                        PRE-SCORING FILTERS                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ Drop         │  │ Age          │  │ Self Tweet   │               │
│  │ Duplicates   │──▶ Filter       │──▶ Filter       │──┐            │
│  └──────────────┘  └──────────────┘  └──────────────┘  │            │
│                                                         │            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │            │
│  │ Previously   │  │ Previously   │  │ Blocked/     │◀─┘            │
│  │ Seen Posts   │◀─│ Served Posts │◀─│ Muted Users  │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
│         │                                                            │
│         ▼                                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ Muted        │  │ Retweet      │  │ Subscription │               │
│  │ Keywords     │──▶ Deduplication│──▶ Filter       │──▶ To Scoring │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
└──────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                     [ SCORING & SELECTION ]
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       POST-SELECTION FILTERS                         │
│  ┌──────────────────────┐        ┌──────────────────────┐           │
│  │ Visibility Filter    │───────▶│ Conversation Dedup   │──▶ Feed   │
│  │ (Safety Rules)       │        │                      │           │
│  └──────────────────────┘        └──────────────────────┘           │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Pre-Scoring Filters

These filters run **before** the ML model scores candidates, reducing the pool of posts that need expensive computation.

### 1. Drop Duplicates Filter

**Purpose**: Remove duplicate post IDs from the candidate pool.

**How it works**:
- Maintains a `HashSet` of seen tweet IDs
- Iterates through candidates sequentially
- Keeps the **first occurrence** of each tweet ID
- Removes subsequent duplicates

**Why duplicates occur**:
- Same post retrieved from both Thunder (in-network) and Phoenix (out-of-network)
- Post appears in multiple retrieval batches
- Rehydration or caching artifacts

```
Input:  [Post A, Post B, Post A, Post C, Post B]
Output: [Post A, Post B, Post C]
Removed: [Post A (dup), Post B (dup)]
```

---

### 2. Age Filter

**Purpose**: Remove posts older than a configured threshold.

**How it works**:
- Extracts timestamp from tweet ID using [Snowflake ID format](https://en.wikipedia.org/wiki/Snowflake_ID)
- Compares post age against `max_age` threshold
- Posts older than threshold are removed

**Snowflake ID timestamp extraction**:
```
Tweet ID (64-bit):
┌─────────────────────────────────────────────────────┐
│ Timestamp (41 bits) │ Datacenter │ Worker │ Sequence│
└─────────────────────────────────────────────────────┘
        ↓
   Milliseconds since X epoch (Nov 4, 2010)
        ↓
   Calculate age = now - creation_time
```

**Why age matters**:
- Old posts are less relevant for real-time feeds
- Prevents stale content from dominating
- Keeps feed fresh and timely

---

### 3. Self Tweet Filter

**Purpose**: Remove posts authored by the requesting user.

**How it works**:
- Compares `candidate.author_id` with `query.user_id`
- If they match, the post is filtered out

**Why filter your own posts**:
- Users don't need to see their own content recommended back to them
- Own posts are accessible via profile
- Feed space is better used for discovery

---

### 4. Previously Seen Posts Filter

**Purpose**: Remove posts the user has already viewed.

**How it works**:
- Uses a **two-layer detection system**:
  1. **Exact match**: Checks against `query.seen_ids` (explicit list from client)
  2. **Bloom filter**: Probabilistic check for posts seen in past sessions

```
                    Previously Seen Detection
┌────────────────────────────────────────────────────────┐
│                                                        │
│   Post ID ──┬──▶ Exact Match (seen_ids set)           │
│             │         │                                │
│             │         ▼                                │
│             │    Found? ──Yes──▶ REMOVE                │
│             │         │                                │
│             │         No                               │
│             │         ▼                                │
│             └──▶ Bloom Filter Check                    │
│                       │                                │
│                       ▼                                │
│                  May contain? ──Yes──▶ REMOVE          │
│                       │                                │
│                       No                               │
│                       ▼                                │
│                     KEEP                               │
└────────────────────────────────────────────────────────┘
```

**Related post IDs checked**:
- The post itself
- Retweeted post ID (if retweet)
- In-reply-to post ID (if reply)

**Bloom filter properties**:
- Space-efficient probabilistic data structure
- False positives possible (may filter unseen posts)
- No false negatives (seen posts always detected)

---

### 5. Previously Served Posts Filter

**Purpose**: Remove posts already served in the current session.

**How it works**:
- Only enabled for **bottom requests** (scrolling down for more content)
- Checks against `query.served_ids` from the session
- Prevents re-serving the same posts while scrolling

**Session context**:
```
Request Type    │ Filter Enabled?
────────────────┼─────────────────
Initial load    │ No
Pull to refresh │ No
Scroll down     │ Yes ✓
```

**Why distinguish from "seen"**:
- "Seen" = user actually viewed/engaged with post
- "Served" = post was delivered to client (may not have been viewed)
- Prevents immediate re-delivery during scroll sessions

---

### 6. Author Socialgraph Filter (Blocked/Muted Accounts)

**Purpose**: Remove posts from users you've blocked or muted.

**How it works**:
- Loads user's `blocked_user_ids` and `muted_user_ids` from features
- Checks each candidate's `author_id` against both lists
- If author is in either list, post is removed

```rust
// Simplified logic
for candidate in candidates {
    let blocked = blocked_user_ids.contains(author_id);
    let muted = muted_user_ids.contains(author_id);

    if blocked || muted {
        removed.push(candidate);
    } else {
        kept.push(candidate);
    }
}
```

**Block vs Mute behavior**:
| Action | Effect in Feed |
|--------|----------------|
| Block | Complete removal |
| Mute | Complete removal |

Both actions have **identical filtering behavior** in the For You feed.

---

### 7. Muted Keyword Filter

**Purpose**: Remove posts containing words/phrases you've muted.

**How it works**:
1. Tokenizes user's muted keywords using `TweetTokenizer`
2. Creates a `UserMutes` matcher with token sequences
3. For each candidate:
   - Tokenizes the post text
   - Checks for matches against muted patterns
   - Removes posts with matches

**Tokenization process**:
```
User mutes: ["crypto scam", "NFT"]
              ↓
         Tokenize
              ↓
Token sequences: [["crypto", "scam"], ["nft"]]
              ↓
         Build matcher
              ↓
Post: "Check out this crypto scam!"
              ↓
         Tokenize: ["check", "out", "this", "crypto", "scam"]
              ↓
         Match found: "crypto scam" → REMOVE
```

**Features**:
- Case-insensitive matching
- Handles multi-word phrases
- Tokenization normalizes text for robust matching

---

### 8. Retweet Deduplication Filter

**Purpose**: Prevent seeing the same content multiple times (as original or retweet).

**How it works**:
- Maintains a `HashSet` of seen content IDs
- For each candidate:
  - **If retweet**: Check if original tweet ID already seen
  - **If original**: Add to seen set (so future retweets are filtered)

```
Processing order matters:

Scenario A: Original comes first
┌────────────────────────────────────────────────────┐
│ 1. Original Post X → KEEP, mark X as seen         │
│ 2. User A retweets X → REMOVE (X already seen)    │
│ 3. User B retweets X → REMOVE (X already seen)    │
└────────────────────────────────────────────────────┘

Scenario B: Retweet comes first
┌────────────────────────────────────────────────────┐
│ 1. User A retweets X → KEEP, mark X as seen       │
│ 2. Original Post X → KEEP (original always kept)  │
│ 3. User B retweets X → REMOVE (X already seen)    │
└────────────────────────────────────────────────────┘
```

**Why this matters**:
- Popular posts get retweeted many times
- Without this filter, feed could be dominated by same content
- Preserves content diversity

---

### 9. Ineligible Subscription Filter

**Purpose**: Remove paywalled content the user can't access.

**How it works**:
- Checks if post is from a subscription-only author
- Verifies if user has an active subscription
- Removes posts from subscription authors if user is not subscribed

```
Post has subscription_author_id?
         │
         ├── No → KEEP
         │
         └── Yes → Check user.subscribed_user_ids
                         │
                         ├── Contains author → KEEP
                         │
                         └── Doesn't contain → REMOVE
```

---

## Post-Selection Filters

These filters run **after** scoring and selection, applying final safety and quality checks.

### 10. Visibility Filter (VF) - Safety Rules

**Purpose**: Enforce safety policies and remove harmful content.

**How it works**:
1. **Hydration phase**: Fetches visibility results from VF service
2. **Filter phase**: Removes posts with `Action::Drop` verdict

**Safety levels by post type**:
| Post Type | Safety Level |
|-----------|--------------|
| In-network (from followed accounts) | `TimelineHome` |
| Out-of-network (recommended) | `TimelineHomeRecommendations` |

Out-of-network posts face **stricter** safety scrutiny since users haven't explicitly chosen to follow these authors.

**Content categories filtered**:
- Deleted posts
- Spam and manipulation
- Violence and threats
- Gore and graphic content
- Harassment
- Hate speech
- Self-harm content
- Child safety violations
- Other policy violations

**VF decision flow**:
```
┌──────────────────────────────────────────────────────┐
│           Visibility Filtering Service               │
│                                                      │
│  Post ──▶ Policy Engine ──▶ Action Decision         │
│                                │                     │
│                                ▼                     │
│                    ┌───────────────────┐             │
│                    │ Action::Drop(...)  │──▶ REMOVE  │
│                    ├───────────────────┤             │
│                    │ Action::Allow      │──▶ KEEP    │
│                    ├───────────────────┤             │
│                    │ Action::Interstitial│──▶ KEEP*  │
│                    └───────────────────┘             │
│                    (* with warning UI)               │
└──────────────────────────────────────────────────────┘
```

---

### 11. Conversation Deduplication Filter

**Purpose**: Limit posts from the same conversation thread.

**How it works**:
- Groups posts by conversation ID (derived from ancestor chain)
- Keeps only the **highest-scored** post per conversation
- Removes lower-scored posts from same thread

**Conversation ID determination**:
```rust
fn get_conversation_id(candidate: &PostCandidate) -> u64 {
    candidate.ancestors
        .iter()
        .copied()
        .min()  // Root of conversation tree
        .unwrap_or(candidate.tweet_id)  // Or self if no ancestors
}
```

**Example**:
```
Conversation Thread:
    Original Post (ID: 100)
        └── Reply A (score: 0.8) ─┬─ ancestors: [100]
        └── Reply B (score: 0.5) ─┤  ancestors: [100]
        └── Reply C (score: 0.9) ─┘  ancestors: [100]

Conversation ID for all: 100 (min ancestor)

Result: Only Reply C (score: 0.9) is kept
```

**Why limit conversation posts**:
- Prevents thread spam in feed
- Encourages content diversity
- Shows most engaging part of conversation

---

## Filter Execution Order

Filters execute **sequentially** in a specific order for correctness and efficiency:

```
1.  DropDuplicatesFilter        // Remove exact duplicates first
2.  CoreDataHydrationFilter     // Remove posts with missing metadata
3.  AgeFilter                   // Remove old posts
4.  SelfTweetFilter             // Remove user's own posts
5.  RetweetDeduplicationFilter  // Dedupe retweets
6.  IneligibleSubscriptionFilter // Remove inaccessible paid content
7.  PreviouslySeenPostsFilter   // Remove already-viewed posts
8.  PreviouslyServedPostsFilter // Remove recently-served posts
9.  MutedKeywordFilter          // Remove muted keyword matches
10. AuthorSocialgraphFilter     // Remove blocked/muted authors

    [ SCORING HAPPENS HERE ]

11. VFFilter                    // Safety enforcement
12. DedupConversationFilter     // Limit conversation posts
```

**Order rationale**:
- Cheap filters run first (duplicates, age)
- Expensive filters run later (keyword matching)
- Safety filters run last (after scoring invests compute)

---

## User Data Used for Filtering

The following user data is fetched during query hydration to power filters:

```rust
pub struct UserFeatures {
    pub muted_keywords: Vec<String>,      // For MutedKeywordFilter
    pub blocked_user_ids: Vec<i64>,       // For AuthorSocialgraphFilter
    pub muted_user_ids: Vec<i64>,         // For AuthorSocialgraphFilter
    pub followed_user_ids: Vec<i64>,      // For in-network detection
    pub subscribed_user_ids: Vec<i64>,    // For IneligibleSubscriptionFilter
}
```

---

## Filter Statistics and Monitoring

Each filter tracks metrics for monitoring and debugging:

| Metric | Description |
|--------|-------------|
| Candidates in | Number of posts entering filter |
| Candidates kept | Number of posts passing filter |
| Candidates removed | Number of posts filtered out |
| Filter latency | Time to execute filter |

This allows the team to:
- Monitor filter health
- Detect anomalies (sudden spike in removals)
- Tune filter aggressiveness
- Debug user-reported issues

---

## Summary

The filtering system ensures your For You feed:

- **Stays fresh** - Old posts are removed
- **Avoids repetition** - Duplicates and seen posts are filtered
- **Respects privacy** - Your own posts aren't recommended back
- **Honors preferences** - Blocked/muted accounts and keywords are enforced
- **Maintains safety** - Harmful content is removed before serving
- **Preserves diversity** - Conversation and retweet spam is limited

All filters work together to create a clean, relevant candidate pool before the ML model scores and ranks content for your personalized feed.
