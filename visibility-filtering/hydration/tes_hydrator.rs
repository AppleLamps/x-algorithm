use crate::hydration::batch::{Hydrated, HydrationBatch, TweetHydrationBatch};
use crate::hydration::fallback_cache::FallbackCache;
use crate::hydration::metrics::{record_batch_size, timed_results};
use crate::models::{
    CoreFeature, MediaFeature, NsfwFeature, TweetCandidateInput, TweetFeatures, TweetId,
};
use crate::rules::SafetyLevel;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use xai_core_entities::entities::{EditControl, MediaEntities, PureCoreData, TakedownReason};
use xai_core_entities::tweet_entity_service_client::TESClient;

const CLIENT_TIMEOUT: Duration = crate::hydration::HYDRATION_TIMEOUT;
const CLIENT: &str = "tes";

pub(crate) type AuthorIdFallbackCache = FallbackCache<TweetId, u64>;

pub struct TesHydrator {
    pub tes_client: Arc<dyn TESClient + Send + Sync>,
    author_id_cache: Option<AuthorIdFallbackCache>,
}

#[derive(Default)]
pub(crate) struct PureCoreHydration {
    pub(crate) core: HashMap<TweetId, PureCoreData>,
    pub(crate) recovered_authors: HashMap<TweetId, u64>,
}

#[derive(Default)]
pub(crate) struct TweetHydration {
    pub(crate) nullcast: TweetHydrationBatch<bool>,
    pub(crate) community: TweetHydrationBatch<i64>,
    pub(crate) nsfw_user: TweetHydrationBatch<bool>,
    pub(crate) nsfw_admin: TweetHydrationBatch<bool>,
    pub(crate) takedown_reasons: TweetHydrationBatch<Vec<TakedownReason>>,
    pub(crate) edit_control: TweetHydrationBatch<EditControl>,
    pub(crate) media: TweetHydrationBatch<MediaFeature>,
}

impl TesHydrator {
    pub(crate) fn new(
        tes_client: Arc<dyn TESClient + Send + Sync>,
        author_id_cache: Option<AuthorIdFallbackCache>,
    ) -> Self {
        Self {
            tes_client,
            author_id_cache,
        }
    }

    pub(crate) fn author_id_fallback_cache(capacity: usize) -> AuthorIdFallbackCache {
        FallbackCache::new("author_id", capacity)
    }

    pub(crate) async fn fetch_pure_core(
        &self,
        tweet_ids: &[TweetId],
        safety_level: SafetyLevel,
    ) -> PureCoreHydration {
        if tweet_ids.is_empty() {
            return PureCoreHydration::default();
        }
        let cache_request = self
            .author_id_cache
            .as_ref()
            .map(|cache| (cache, cache.begin_request()));
        let candidate_count_by_key = candidates_per_tweet(tweet_ids);
        let raw_ids: Vec<u64> = candidate_count_by_key.keys().copied().collect();
        record_batch_size(CLIENT, candidate_count_by_key.len());
        let fetched = timed_results(
            CLIENT,
            "get_tweet_core_datas",
            safety_level,
            &candidate_count_by_key,
            CLIENT_TIMEOUT,
            self.tes_client.get_tweet_core_datas(raw_ids),
        )
        .await;
        resolve_pure_core(cache_request, fetched.map_keys(TweetId))
    }

    pub(crate) async fn hydrate_tweets(
        &self,
        tweet_ids: &[TweetId],
        safety_level: SafetyLevel,
    ) -> TweetHydration {
        let candidate_count_by_key = candidates_per_tweet(tweet_ids);
        let raw_ids: Vec<u64> = candidate_count_by_key.keys().copied().collect();

        let (
            nullcast,
            community,
            nsfw_user,
            nsfw_admin,
            takedown_reasons,
            edit_control,
            media_entities,
        ) = tokio::join!(
            timed_results(
                CLIENT,
                "get_nullcast",
                safety_level,
                &candidate_count_by_key,
                CLIENT_TIMEOUT,
                self.tes_client.get_nullcast(raw_ids.clone()),
            ),
            timed_results(
                CLIENT,
                "get_community",
                safety_level,
                &candidate_count_by_key,
                CLIENT_TIMEOUT,
                self.tes_client.get_community(raw_ids.clone()),
            ),
            timed_results(
                CLIENT,
                "get_nsfw_user",
                safety_level,
                &candidate_count_by_key,
                CLIENT_TIMEOUT,
                self.tes_client.get_nsfw_user(raw_ids.clone()),
            ),
            timed_results(
                CLIENT,
                "get_nsfw_admin",
                safety_level,
                &candidate_count_by_key,
                CLIENT_TIMEOUT,
                self.tes_client.get_nsfw_admin(raw_ids.clone()),
            ),
            timed_results(
                CLIENT,
                "get_takedown_reasons",
                safety_level,
                &candidate_count_by_key,
                CLIENT_TIMEOUT,
                self.tes_client.get_takedown_reasons(raw_ids.clone()),
            ),
            timed_results(
                CLIENT,
                "get_edit_control",
                safety_level,
                &candidate_count_by_key,
                CLIENT_TIMEOUT,
                self.tes_client.get_edit_control(raw_ids.clone()),
            ),
            timed_results(
                CLIENT,
                "get_tweet_media_entities",
                safety_level,
                &candidate_count_by_key,
                CLIENT_TIMEOUT,
                self.tes_client.get_tweet_media_entities(raw_ids.clone()),
            ),
        );

        TweetHydration {
            nullcast: nullcast.map_keys(TweetId),
            community: community.map_keys(TweetId),
            nsfw_user: nsfw_user.map_keys(TweetId),
            nsfw_admin: nsfw_admin.map_keys(TweetId),
            takedown_reasons: takedown_reasons.map_keys(TweetId),
            edit_control: edit_control.map_keys(TweetId),
            media: media_entities.map_keys(TweetId).map(media_feature),
        }
    }

    pub(crate) fn assemble_tweet_features(
        &self,
        candidates: &[TweetCandidateInput],
        core_datas: &HashMap<TweetId, PureCoreData>,
        tweet_keyed: &TweetHydration,
    ) -> HashMap<TweetId, TweetFeatures> {
        candidates
            .iter()
            .map(|c| {
                (
                    c.tweet_id,
                    build_tweet_features(c.tweet_id, core_datas, tweet_keyed),
                )
            })
            .collect()
    }
}

fn candidates_per_tweet(tweet_ids: &[TweetId]) -> HashMap<u64, usize> {
    let mut candidate_count_by_key = HashMap::with_capacity(tweet_ids.len());
    for tweet_id in tweet_ids {
        *candidate_count_by_key.entry(tweet_id.0).or_default() += 1;
    }
    candidate_count_by_key
}

fn resolve_pure_core(
    cache_request: Option<(&AuthorIdFallbackCache, u64)>,
    fetched: TweetHydrationBatch<PureCoreData>,
) -> PureCoreHydration {
    let mut core = HashMap::new();
    let mut author_ids = HashMap::new();
    for (tweet_id, hydrated) in fetched.into_hydrated() {
        let author_id = match hydrated {
            Hydrated::Found(core_data) => {
                let author_id = core_data.author_id;
                core.insert(tweet_id, core_data);
                Hydrated::Found(author_id)
            }
            Hydrated::NotFound => Hydrated::NotFound,
            Hydrated::Failed(error) => Hydrated::Failed(error),
        };
        author_ids.insert(tweet_id, author_id);
    }
    let Some((cache, generation)) = cache_request else {
        return PureCoreHydration {
            core,
            recovered_authors: HashMap::new(),
        };
    };
    let recovered_authors = cache
        .resolve_hydration_batch(generation, HydrationBatch::from_hydrated(author_ids))
        .into_hydrated()
        .into_iter()
        .filter_map(|(tweet_id, hydrated)| match hydrated {
            Hydrated::Found(author_id) if !core.contains_key(&tweet_id) => {
                Some((tweet_id, author_id))
            }
            _ => None,
        })
        .collect();
    PureCoreHydration {
        core,
        recovered_authors,
    }
}

fn build_tweet_features(
    tweet_id: TweetId,
    core_datas: &HashMap<TweetId, PureCoreData>,
    tweet_keyed: &TweetHydration,
) -> TweetFeatures {
    let id = tweet_id;

    let media = tweet_keyed.media.get_or_default(&id);
    let is_nullcast = tweet_keyed.nullcast.get(&id).copied().unwrap_or(false);
    let is_community_tweet = tweet_keyed.community.get(&id).is_some();
    let takedown_reasons = tweet_keyed.takedown_reasons.get_or_default(&id);
    let nsfw = NsfwFeature {
        user: tweet_keyed.nsfw_user.get(&id).copied().unwrap_or(false),
        admin: tweet_keyed.nsfw_admin.get(&id).copied().unwrap_or(false),
    };
    let edit_control = tweet_keyed.edit_control.get(&id).cloned();

    let core = core_datas
        .get(&tweet_id)
        .map(|core_data| CoreFeature {
            text: core_data.text.clone(),
            source_tweet_id: core_data.source_tweet_id,
        })
        .unwrap_or_default();

    TweetFeatures {
        core,
        media,
        takedown_reasons,
        nsfw,
        is_nullcast,
        is_community_tweet,
        edit_control,
    }
}

fn media_feature(entities: MediaEntities) -> MediaFeature {
    let mut feature = MediaFeature {
        has_media: !entities.is_empty(),
        ..Default::default()
    };

    for restrictions in entities
        .iter()
        .filter(|e| e.media_key.is_some())
        .filter_map(|e| e.additional_metadata.as_ref())
        .filter_map(|metadata| metadata.restrictions.as_ref())
    {
        feature.has_dmca_media |= restrictions.is_dmca == Some(true);
        if let Some(geo) = &restrictions.geo_restrictions {
            feature
                .geo_allow_list
                .extend(geo.whitelisted_country_codes.iter().flatten().cloned());
            feature
                .geo_deny_list
                .extend(geo.blacklisted_country_codes.iter().flatten().cloned());
        }
    }

    feature
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hydration::batch::HydrationError;
    use crate::models::{resolve_candidate, RawCandidate};
    use xai_core_entities::entities::{MediaEntity, PureCoreData};
    use xai_core_entities::tweet_entity_service_client::MockTESClient;
    use xai_x_thrift::media_information::{AdditionalMetadata, Restrictions};

    fn found<V>(id: u64, value: V) -> TweetHydrationBatch<V> {
        TweetHydrationBatch::from_results(
            [TweetId(id)],
            HashMap::from([(TweetId(id), Ok::<_, anyhow::Error>(Some(value)))]),
        )
    }

    fn candidate(tweet_id: u64, author_id: u64) -> TweetCandidateInput {
        let core = HashMap::from([(
            TweetId(tweet_id),
            PureCoreData {
                author_id,
                ..Default::default()
            },
        )]);
        resolve_candidate(
            &RawCandidate {
                tweet_id: TweetId(tweet_id),
                request_author_id: None,
            },
            &core,
            &HashMap::new(),
        )
        .unwrap()
    }

    fn hydrator() -> TesHydrator {
        TesHydrator::new(Arc::new(MockTESClient::default()), None)
    }

    #[test]
    fn tweet_candidate_counts_deduplicate_backend_keys() {
        let tweet_ids = vec![TweetId(1), TweetId(1), TweetId(2)];

        assert_eq!(
            candidates_per_tweet(&tweet_ids),
            HashMap::from([(1, 2), (2, 1)])
        );
    }

    fn core_batch(
        entries: impl IntoIterator<Item = (u64, Hydrated<u64>)>,
    ) -> TweetHydrationBatch<PureCoreData> {
        HydrationBatch::from_hydrated(
            entries
                .into_iter()
                .map(|(id, hydrated)| {
                    let hydrated = match hydrated {
                        Hydrated::Found(author_id) => Hydrated::Found(PureCoreData {
                            author_id,
                            ..Default::default()
                        }),
                        Hydrated::NotFound => Hydrated::NotFound,
                        Hydrated::Failed(e) => Hydrated::Failed(e),
                    };
                    (TweetId(id), hydrated)
                })
                .collect(),
        )
    }

    fn failed() -> Hydrated<u64> {
        Hydrated::Failed(HydrationError::Timeout)
    }

    fn author_id_cache() -> AuthorIdFallbackCache {
        TesHydrator::author_id_fallback_cache(8)
    }

    #[test]
    fn failed_pure_core_recovers_only_previously_found_author_ids() {
        let cache = author_id_cache();
        let first = resolve_pure_core(
            Some((&cache, cache.begin_request())),
            core_batch([(1, Hydrated::Found(10)), (2, Hydrated::NotFound)]),
        );
        assert!(first.recovered_authors.is_empty());

        let second = resolve_pure_core(
            Some((&cache, cache.begin_request())),
            core_batch([(1, failed()), (2, failed()), (3, failed())]),
        );

        assert!(second.core.is_empty());
        assert_eq!(second.recovered_authors, HashMap::from([(TweetId(1), 10)]));

        let fresh_again = resolve_pure_core(
            Some((&cache, cache.begin_request())),
            core_batch([(1, Hydrated::Found(10))]),
        );
        assert!(fresh_again.recovered_authors.is_empty());
    }

    #[test]
    fn without_cache_failed_pure_core_recovers_nothing() {
        resolve_pure_core(None, core_batch([(1, Hydrated::Found(10))]));

        let second = resolve_pure_core(None, core_batch([(1, failed())]));

        assert!(second.core.is_empty());
        assert!(second.recovered_authors.is_empty());
    }

    fn dmca_media_entity(has_media_key: bool) -> MediaEntity {
        MediaEntity {
            media_key: has_media_key.then(Default::default),
            additional_metadata: Some(AdditionalMetadata {
                restrictions: Some(Restrictions {
                    is_dmca: Some(true),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    #[test]
    fn assemble_flattens_media_geo_restrictions_from_tes() {
        use xai_core_entities::entities::MediaEntity;
        use xai_x_thrift::media_information::{AdditionalMetadata, GeoRestrictions, Restrictions};
        let candidates = vec![candidate(10, 100)];
        let core_datas = HashMap::from([(
            TweetId(10),
            PureCoreData {
                author_id: 100,
                ..Default::default()
            },
        )]);
        let entity = |has_media_key: bool, allow: &[&str], deny: &[&str]| MediaEntity {
            media_key: has_media_key.then(Default::default),
            additional_metadata: Some(AdditionalMetadata {
                restrictions: Some(Restrictions {
                    is_dmca: Some(false),
                    geo_restrictions: Some(GeoRestrictions {
                        whitelisted_country_codes: Some(
                            allow.iter().map(|s| s.to_string()).collect(),
                        ),
                        blacklisted_country_codes: Some(
                            deny.iter().map(|s| s.to_string()).collect(),
                        ),
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        };
        let tweet_keyed = TweetHydration {
            media: found(
                10,
                media_feature(vec![
                    entity(true, &["us"], &["de"]),
                    entity(true, &["gb"], &["fr"]),
                    entity(false, &["ignored"], &["ignored"]),
                    MediaEntity::default(),
                ]),
            ),
            ..Default::default()
        };

        let features = hydrator().assemble_tweet_features(&candidates, &core_datas, &tweet_keyed);

        let f = &features[&TweetId(10)];
        assert_eq!(f.media.geo_allow_list, vec!["us", "gb"]);
        assert_eq!(f.media.geo_deny_list, vec!["de", "fr"]);
    }

    #[test]
    fn assemble_defaults_geo_lists_when_media_entities_missing() {
        let candidates = vec![candidate(10, 100)];
        let core_datas = HashMap::from([(
            TweetId(10),
            PureCoreData {
                author_id: 100,
                ..Default::default()
            },
        )]);

        let features = hydrator().assemble_tweet_features(
            &candidates,
            &core_datas,
            &TweetHydration::default(),
        );

        let f = &features[&TweetId(10)];
        assert!(f.media.geo_allow_list.is_empty());
        assert!(f.media.geo_deny_list.is_empty());
    }

    #[test]
    fn assemble_hydrates_dmca_media() {
        let candidates = vec![candidate(10, 100)];
        let core_datas = HashMap::from([(
            TweetId(10),
            PureCoreData {
                author_id: 100,
                ..Default::default()
            },
        )]);
        let tweet_keyed = TweetHydration {
            media: found(10, media_feature(vec![dmca_media_entity(true)])),
            ..Default::default()
        };

        let features = hydrator().assemble_tweet_features(&candidates, &core_datas, &tweet_keyed);

        assert!(features[&TweetId(10)].media.has_dmca_media);
        assert!(features[&TweetId(10)].media.has_media);
    }

    #[test]
    fn assemble_derives_has_media_from_media_entities() {
        let candidates = vec![candidate(10, 100), candidate(11, 100)];
        let core_datas = HashMap::from([
            (
                TweetId(10),
                PureCoreData {
                    author_id: 100,
                    ..Default::default()
                },
            ),
            (
                TweetId(11),
                PureCoreData {
                    author_id: 100,
                    ..Default::default()
                },
            ),
        ]);
        let tweet_keyed = TweetHydration {
            media: TweetHydrationBatch::from_results(
                [TweetId(10), TweetId(11)],
                HashMap::from([
                    (
                        TweetId(10),
                        Ok::<_, anyhow::Error>(Some(media_feature(vec![MediaEntity::default()]))),
                    ),
                    (
                        TweetId(11),
                        Ok::<_, anyhow::Error>(Some(media_feature(Vec::<MediaEntity>::new()))),
                    ),
                ]),
            ),
            ..Default::default()
        };

        let features = hydrator().assemble_tweet_features(&candidates, &core_datas, &tweet_keyed);

        assert!(features[&TweetId(10)].media.has_media);
        assert!(!features[&TweetId(11)].media.has_media);
    }

    #[test]
    fn dmca_metadata_without_media_key_is_ignored() {
        let feature = media_feature(vec![dmca_media_entity(false)]);
        assert!(!feature.has_dmca_media);
    }

    #[test]
    fn assemble_hydrates_text_from_core_data() {
        let candidates = vec![candidate(10, 100)];
        let core_datas = HashMap::from([(
            TweetId(10),
            PureCoreData {
                author_id: 100,
                text: "muted words".to_string(),
                ..Default::default()
            },
        )]);

        let features = hydrator().assemble_tweet_features(
            &candidates,
            &core_datas,
            &TweetHydration::default(),
        );

        assert_eq!(features[&TweetId(10)].core.text, "muted words");
    }

    #[test]
    fn assemble_preserves_independent_features_when_core_missing() {
        let candidates = vec![candidate(10, 100)];
        let tweet_keyed = TweetHydration {
            nullcast: found(10, true),
            community: found(10, 1),
            nsfw_user: found(10, true),
            nsfw_admin: found(10, true),
            takedown_reasons: found(10, vec![TakedownReason::Dmca]),
            edit_control: found(10, EditControl::Initial(Default::default())),
            media: found(10, media_feature(vec![dmca_media_entity(true)])),
        };

        let features =
            hydrator().assemble_tweet_features(&candidates, &HashMap::new(), &tweet_keyed);

        let f = &features[&TweetId(10)];
        assert!(f.core.text.is_empty());
        assert_eq!(f.core.source_tweet_id, None);
        assert!(f.is_nullcast);
        assert!(f.is_community_tweet);
        assert!(f.nsfw.user && f.nsfw.admin);
        assert_eq!(f.takedown_reasons, vec![TakedownReason::Dmca]);
        assert!(f.edit_control.is_some());
        assert!(f.media.has_media && f.media.has_dmca_media);
    }

    #[test]
    fn missing_core_keeps_nullcast_drop_for_supplied_and_recovered_authors() {
        use crate::models::{HydratedTweetCandidate, VfAction, ViewerFeatures};
        use crate::rules::RuleEngine;

        let cache = author_id_cache();
        resolve_pure_core(
            Some((&cache, cache.begin_request())),
            core_batch([(10, Hydrated::Found(100))]),
        );
        let pure_core = resolve_pure_core(
            Some((&cache, cache.begin_request())),
            core_batch([(10, failed())]),
        );
        let engine = RuleEngine::for_tests();
        for request_author_id in [None, Some(100)] {
            let candidate = resolve_candidate(
                &RawCandidate {
                    tweet_id: TweetId(10),
                    request_author_id,
                },
                &pure_core.core,
                &pure_core.recovered_authors,
            )
            .unwrap();
            for is_nullcast in [false, true] {
                let features = hydrator().assemble_tweet_features(
                    &[candidate],
                    &pure_core.core,
                    &TweetHydration {
                        nullcast: found(10, is_nullcast),
                        ..Default::default()
                    },
                );
                let hydrated = HydratedTweetCandidate {
                    tweet_id: candidate.tweet_id.0,
                    author_id: candidate.author_id.get(),
                    tweet_features: features[&candidate.tweet_id].clone(),
                    ..Default::default()
                };
                for level in [
                    SafetyLevel::TimelineHome,
                    SafetyLevel::TimelineHomeRecommendations,
                ] {
                    let verdict = engine.evaluate(level, &ViewerFeatures::default(), &hydrated);
                    if is_nullcast {
                        assert!(matches!(verdict.action, VfAction::Drop(_)));
                        assert_eq!(verdict.decided_by, Some("NullcastedTweetDropRule"));
                    } else {
                        assert!(matches!(verdict.action, VfAction::Allow));
                    }
                }
            }
        }
    }
}
