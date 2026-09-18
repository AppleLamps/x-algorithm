use crate::clients::socialgraph_client::SocialgraphClient;
use crate::hydration::metrics::{batch_outcome, record_batch_size, timed_rpc, HydratorOutcome};
use crate::models::{ExclusiveContentFeatures, TweetId, Viewer};
use crate::rules::SafetyLevel;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use xai_core_entities::tweet_entity_service_client::TESClient;

const CLIENT_TIMEOUT: Duration = crate::hydration::HYDRATION_TIMEOUT;
const CLIENT: &str = "exclusive_content";

pub struct ExclusiveContentHydrator {
    pub tes_client: Arc<dyn TESClient + Send + Sync>,
    pub sg_client: Arc<dyn SocialgraphClient + Send + Sync>,
}

impl ExclusiveContentHydrator {
    pub async fn hydrate(
        &self,
        tweet_ids: &[TweetId],
        viewer: Viewer,
        safety_level: SafetyLevel,
    ) -> HashMap<TweetId, Option<ExclusiveContentFeatures>> {
        let raw_ids: Vec<u64> = tweet_ids.iter().map(|t| t.0).collect();
        let candidate_count = raw_ids.len();
        record_batch_size(CLIENT, candidate_count);

        let exclusive_controls = timed_rpc(
            CLIENT,
            "get_exclusive_controls",
            safety_level,
            candidate_count,
            CLIENT_TIMEOUT,
            batch_outcome,
            self.tes_client.get_exclusive_controls(raw_ids),
        )
        .await;

        let root_author_ids: Vec<u64> = exclusive_controls
            .values()
            .filter_map(|r| {
                r.as_ref()
                    .ok()
                    .and_then(|opt| opt.as_ref())
                    .map(|ctrl| ctrl.conversation_author_id)
            })
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        let super_follows = match viewer.user_id() {
            Some(vid) if !root_author_ids.is_empty() => {
                timed_rpc(
                    CLIENT,
                    "batch_check_super_follows",
                    safety_level,
                    candidate_count,
                    CLIENT_TIMEOUT,
                    |_| HydratorOutcome::Success,
                    self.sg_client
                        .batch_check_super_follows(vid, &root_author_ids),
                )
                .await
            }
            _ => HashMap::new(),
        };

        tweet_ids
            .iter()
            .map(|tweet_id| {
                let features = exclusive_controls
                    .get(&tweet_id.0)
                    .and_then(|r| r.as_ref().ok())
                    .and_then(|opt| opt.as_ref())
                    .map(|ctrl| {
                        let author_id = ctrl.conversation_author_id;
                        ExclusiveContentFeatures {
                            conversation_author_id: author_id,
                            viewer_super_follows_author: super_follows
                                .get(&author_id)
                                .copied()
                                .unwrap_or(false),
                        }
                    });
                (*tweet_id, features)
            })
            .collect()
    }
}
