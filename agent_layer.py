# -*- coding: utf-8 -*-
"""
Marketing Agent Layer: Decision & Recommendation System
========================================================

Purpose
-------
This module transforms purchase predictions into actionable marketing strategies.
It sits on top of the behavior-to-agent pipeline outputs to generate
marketing insights and recommendations.

Conceptual Flow
---------------
[behavior_to_agent_pipeline.py outputs]
 → [Agent Layer: Segmentation & Action] ← YOU ARE HERE
 → [LLM: Natural Language Explanation & Creative]

Pipeline Flow
-------------
[inference_targets.csv] (purchase probabilities)
    + [service_eta_predictions.csv] (time-to-action estimates)
    → Marketing Agent Layer
    → Segmentation, Prioritization, Action Recommendations
    → [marketing_segments.csv, marketing_actions.json, marketing_report.txt]

Notes
-----
- Works with outputs from behavior_to_agent_pipeline.py
- Generates actionable marketing plans without semantic coupling
- Optional LLM integration for copy generation
"""

from __future__ import annotations

import os
import time
import json
import warnings
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from openai import OpenAI

warnings.filterwarnings("ignore")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class AgentConfig:
    """Marketing Agent Configuration"""
    
    # Input Paths (from behavior_to_agent_pipeline.py)
    prob_predictions_path: str = "inference_targets.csv"
    time_predictions_path: str = "service_eta_predictions.csv"
    
    # Output Paths
    output_segments_path: str = "marketing_segments.csv"
    output_actions_path: str = "marketing_actions.json"
    output_report_path: str = "marketing_report.txt"
    
    # Segmentation Thresholds
    high_prob_threshold: float = 0.7
    medium_prob_threshold: float = 0.4
    urgent_days_threshold: float = 2.0
    medium_days_threshold: float = 5.0
    
    # Business Rules
    min_prob_for_action: float = 0.3
    max_eta_days_actionable: float = 14.0
    
    # Priority Scoring Weights
    weight_probability: float = 0.6
    weight_urgency: float = 0.4
    
    # Campaign Constraints
    max_campaign_users: Optional[int] = None
    min_expected_roi: float = 1.0
    
    # LLM Decision Budget (속도/비용 제어)
    use_llm_top_k: int = 50        # priority 상위 K명만 LLM 최종선택
    use_llm_min_score: float = 70.0   # 또는 priority_score가 이 이상만


# ============================================================================
# Enums
# ============================================================================

class PurchaseUrgency(Enum):
    """Purchase urgency based on time-to-action estimates"""
    CRITICAL = "critical"  # 0-2 days
    HIGH = "high"  # 2-5 days
    MEDIUM = "medium"  # 5-10 days
    LOW = "low"  # 10+ days
    UNCERTAIN = "uncertain"  # No valid prediction


class PurchaseIntent(Enum):
    """Purchase intent based on probability"""
    VERY_HIGH = "very_high"  # 0.7+
    HIGH = "high"  # 0.5-0.7
    MEDIUM = "medium"  # 0.3-0.5
    LOW = "low"  # <0.3


class MarketingAction(Enum):
    """Recommended marketing actions"""
    IMMEDIATE_DISCOUNT = "immediate_discount"
    REMINDER_EMAIL = "reminder_email"
    NURTURE_CAMPAIGN = "nurture_campaign"
    RETARGETING_AD = "retargeting_ad"
    CART_RECOVERY = "cart_recovery"
    NO_ACTION = "no_action"
    MONITOR = "monitor"


# ============================================================================
# Core Agent
# ============================================================================

class MarketingAgentLayer:
    """
    Marketing Agent Layer
    
    Responsibilities:
    1. Load and merge probability and time predictions
    2. Segment users into actionable groups
    3. Calculate priority scores
    4. Recommend specific marketing actions
    5. Generate business insights and reports
    """
    
    def __init__(self, config: AgentConfig, llm_assistant: Optional["LLMMarketingAssistant"] = None):
        self.config = config
        self.llm_assistant = llm_assistant

        self.df_merged = None
        self.segments = None
        self.actions = None

        # export USE_LLM_DECISION="1" 로 켜기 (기본 off)
        self.use_llm_decision = os.getenv("USE_LLM_DECISION", "0") == "1"

        
    def load_and_merge_predictions(self) -> pd.DataFrame:
        """Load probability and time predictions, then merge on user_id + category_id"""
        
        print("=" * 80)
        print("LOADING PREDICTIONS")
        print("=" * 80)
        
        # Load probability predictions
        if not os.path.exists(self.config.prob_predictions_path):
            raise FileNotFoundError(
                f"Probability predictions not found: {self.config.prob_predictions_path}"
            )
        
        df_prob = pd.read_csv(self.config.prob_predictions_path)
        print(f"✓ Probability predictions loaded: {len(df_prob):,} user-category pairs")
        print(f"  Columns: {list(df_prob.columns)}")
        
        # Load time predictions
        if not os.path.exists(self.config.time_predictions_path):
            warnings.warn(
                f"Time predictions not found: {self.config.time_predictions_path}"
            )
            df_time = pd.DataFrame()
        else:
            df_time = pd.read_csv(self.config.time_predictions_path)
            print(f"✓ Time predictions loaded: {len(df_time):,} user-category pairs")
            print(f"  Columns: {list(df_time.columns)}")
        
        # Merge on user_id + category_id
        if len(df_time) > 0:
            df_merged = pd.merge(
                df_prob,
                df_time[['user_id', 'category_id', 'eta_days', 'eta_sec', 
                        'was_clipped', 'n_events']],
                on=['user_id', 'category_id'],
                how='left'
            )
        else:
            df_merged = df_prob.copy()
            df_merged['eta_days'] = np.nan
            df_merged['eta_sec'] = np.nan
            df_merged['was_clipped'] = False
            df_merged['n_events'] = 0
        
        print(f"✓ Merged dataset: {len(df_merged):,} records")
        print()
        
        self.df_merged = df_merged
        return df_merged
    
    def classify_urgency(self, eta_days: float) -> PurchaseUrgency:
        """Classify urgency level based on estimated time-to-action"""
        if pd.isna(eta_days) or eta_days < 0:
            return PurchaseUrgency.UNCERTAIN
        elif eta_days <= self.config.urgent_days_threshold:
            return PurchaseUrgency.CRITICAL
        elif eta_days <= self.config.medium_days_threshold:
            return PurchaseUrgency.HIGH
        elif eta_days <= 10.0:
            return PurchaseUrgency.MEDIUM
        else:
            return PurchaseUrgency.LOW
    
    def classify_intent(self, probability: float) -> PurchaseIntent:
        """Classify purchase intent based on probability"""
        if probability >= self.config.high_prob_threshold:
            return PurchaseIntent.VERY_HIGH
        elif probability >= 0.5:
            return PurchaseIntent.HIGH
        elif probability >= self.config.medium_prob_threshold:
            return PurchaseIntent.MEDIUM
        else:
            return PurchaseIntent.LOW
    
    def calculate_priority_score(self, row: pd.Series) -> float:
        """
        Calculate unified priority score (0-100)
        
        Combines probability and urgency using weighted formula:
        - Higher probability → higher score
        - Shorter time-to-action → higher score
        """
        prob = row['prob']
        eta_days = row.get('eta_days_adj', np.nan)

        
        # Probability component (0-100)
        prob_score = prob * 100
        
        # Urgency component (0-100)
        if pd.isna(eta_days) or eta_days < 0:
            urgency_score = 0
        else:
            max_days = self.config.max_eta_days_actionable
            urgency_score = max(0, 100 * (1 - eta_days / max_days))
        
        # Weighted combination
        final_score = (
            self.config.weight_probability * prob_score +
            self.config.weight_urgency * urgency_score
        )
        
        return round(final_score, 2)
    
    def recommend_action(self, row: pd.Series) -> MarketingAction:
        """
        Recommend specific marketing action based on segment
        
        Decision Logic:
        1. Critical urgency + very high intent → immediate discount
        2. High urgency + high intent → reminder email
        3. Medium intent + short ETA → retargeting ad
        4. Medium intent + long ETA → nurture campaign
        5. Low intent → monitor or no action
        """
        prob = row['prob']
        urgency = row['urgency']
        intent = row['intent']
        
        # Filter by minimum probability
        if prob < self.config.min_prob_for_action:
            return MarketingAction.NO_ACTION
        
        # Critical urgency + very high intent
        if (urgency == PurchaseUrgency.CRITICAL.value and 
            intent == PurchaseIntent.VERY_HIGH.value):
            return MarketingAction.IMMEDIATE_DISCOUNT
        
        # High urgency + high/very high intent
        if (urgency in [PurchaseUrgency.CRITICAL.value, PurchaseUrgency.HIGH.value] and 
            intent in [PurchaseIntent.VERY_HIGH.value, PurchaseIntent.HIGH.value]):
            return MarketingAction.REMINDER_EMAIL
        
        # Medium urgency + high intent
        if (urgency == PurchaseUrgency.MEDIUM.value and 
            intent == PurchaseIntent.HIGH.value):
            return MarketingAction.RETARGETING_AD
        
        # Medium intent
        if intent == PurchaseIntent.MEDIUM.value:
            if urgency in [PurchaseUrgency.CRITICAL.value, PurchaseUrgency.HIGH.value]:
                return MarketingAction.RETARGETING_AD
            else:
                return MarketingAction.NURTURE_CAMPAIGN
        
        return MarketingAction.MONITOR
    
    def _candidate_actions_by_rule(self, row: pd.Series) -> List[str]:
        """
        Rule 기반으로 후보 액션 Top-K 생성.
        LLM은 반드시 이 후보 중에서만 고르게 한다.
        """
        prob = float(row["prob"])
        urgency = row["urgency"]
        intent = row["intent"]

        # 확률이 너무 낮으면 보수적으로
        if prob < self.config.min_prob_for_action:
            return [MarketingAction.NO_ACTION.value, MarketingAction.MONITOR.value]

        candidates: List[str] = []

        # 1) 기존 rule 추천을 1순위로 포함
        primary = self.recommend_action(row).value
        candidates.append(primary)

        # 2) 보완 후보 추가 (현실적인 대안 1~2개)
        # 급하고 의도가 높으면 reminder를 항상 후보에
        if urgency in [PurchaseUrgency.CRITICAL.value, PurchaseUrgency.HIGH.value] and intent in [
            PurchaseIntent.VERY_HIGH.value, PurchaseIntent.HIGH.value
        ]:
            if MarketingAction.REMINDER_EMAIL.value not in candidates:
                candidates.append(MarketingAction.REMINDER_EMAIL.value)

        # 중간 이상 의도면 retargeting 후보
        if intent in [PurchaseIntent.MEDIUM.value, PurchaseIntent.HIGH.value]:
            if MarketingAction.RETARGETING_AD.value not in candidates:
                candidates.append(MarketingAction.RETARGETING_AD.value)

        # 중간 의도 + 덜 급하면 nurture 후보
        if intent == PurchaseIntent.MEDIUM.value and urgency in [
            PurchaseUrgency.MEDIUM.value, PurchaseUrgency.LOW.value, PurchaseUrgency.UNCERTAIN.value
        ]:
            if MarketingAction.NURTURE_CAMPAIGN.value not in candidates:
                candidates.append(MarketingAction.NURTURE_CAMPAIGN.value)

        # 할인은 남발 방지: “매우 급하고 + 매우 높은 의도”일 때만 후보
        if intent == PurchaseIntent.VERY_HIGH.value and urgency == PurchaseUrgency.CRITICAL.value:
            if MarketingAction.IMMEDIATE_DISCOUNT.value not in candidates:
                candidates.append(MarketingAction.IMMEDIATE_DISCOUNT.value)

        # 항상 안전 후보 포함
        if MarketingAction.MONITOR.value not in candidates:
            candidates.append(MarketingAction.MONITOR.value)

        # 중복 제거(순서 유지) + 길이 제한
        uniq = []
        seen = set()
        for a in candidates:
            if a not in seen:
                uniq.append(a)
                seen.add(a)

        return uniq[:4]

    def _llm_choose_action(self, row: pd.Series, candidates: List[str]) -> Optional[Dict[str, Any]]:
        """
        LLM에게 후보 중 최종 액션을 선택하게 한다.
        실패하면 None 반환 → rule fallback
        """
        if not self.use_llm_decision:
            return None
        if self.llm_assistant is None:
            return None

        payload = {
            "user_id": str(row.get("user_id", "")),
            "category_id": str(row.get("category_id", "")),
            "prob": float(row.get("prob", 0.0)),
            "eta_days": None if pd.isna(row.get("eta_days")) else float(row.get("eta_days")),
            "eta_days_adj": None if pd.isna(row.get("eta_days_adj")) else float(row.get("eta_days_adj")),
            "n_events": int(row.get("n_events", 0)),
            "too_short": bool(row.get("too_short", False)) if "too_short" in row else None,
            "intent": str(row.get("intent", "")),
            "urgency": str(row.get("urgency", "")),
            "candidates": candidates,
        }

        try:
            return self.llm_assistant.decide_best_action(payload)
        except Exception as e:
            warnings.warn(f"LLM decision failed; fallback to rule. err={e}")
            return None

    def recommend_action_hybrid(self, row: pd.Series) -> str:
        """
        최종 추천 액션:
        - rule로 후보 생성
        - LLM이 후보 중 1개 선택
        - 실패하면 rule 추천(후보 1순위)로 fallback
        """
        candidates = self._candidate_actions_by_rule(row)

        llm_out = self._llm_choose_action(row, candidates)
        if isinstance(llm_out, dict):
            chosen = llm_out.get("action")
            if chosen in candidates:
                return chosen

        return candidates[0]

    
    def segment_users(self) -> pd.DataFrame:
        """
        Segment users based on predictions

        Output columns:
        - All original columns
        - urgency: PurchaseUrgency classification
        - intent: PurchaseIntent classification
        - priority_score: Unified score (0-100)
        - recommended_action: Hybrid decision (rule baseline + optional LLM on Top-K)
        - segment_name: Human-readable segment
        """

        print("=" * 80)
        print("SEGMENTING USERS")
        print("=" * 80)

        df = self.df_merged.copy()

        # 기존 로직 유지 (너 코드에서 eta_days_adj를 이렇게 만들고 있음)
        df['eta_days_adj'] = np.clip(
            df['eta_days'] * 30,   # scale factor (business decision horizon)
            0,
            self.config.max_eta_days_actionable
        )

        # Classify urgency and intent
        df['urgency'] = df['eta_days_adj'].apply(self.classify_urgency).apply(lambda x: x.value)
        df['intent'] = df['prob'].apply(self.classify_intent).apply(lambda x: x.value)

        # Priority score 먼저 계산 (LLM 대상 선정에 필요)
        df['priority_score'] = df.apply(self.calculate_priority_score, axis=1)

        # 1) 우선 rule 기반으로 recommended_action 채우기 (빠름)
        df['recommended_action'] = df.apply(
            lambda row: self.recommend_action(row).value,
            axis=1
        )

        # 2) LLM 최종 선택은 일부에만 적용 (Top-K + min_score)
        use_llm = bool(getattr(self, "use_llm_decision", False)) and (self.llm_assistant is not None)
        if use_llm:
            # priority 높은 순으로 정렬해서 상위부터 LLM 적용
            df = df.sort_values('priority_score', ascending=False).reset_index(drop=True)

            # min_score 조건
            mask = (df['priority_score'] >= float(self.config.use_llm_min_score))

            # Top-K 제한
            top_k = int(self.config.use_llm_top_k) if self.config.use_llm_top_k is not None else None
            idx = df.index[mask]
            if top_k is not None:
                idx = idx[:top_k]

            print(f"✓ LLM decision enabled: applying to {len(idx):,} rows (top_k={top_k}, min_score={self.config.use_llm_min_score})")

            # 루프에서 LLM 호출 (최대 top_k번만)
            # - recommend_action_hybrid 내부에서 후보 생성 + LLM 선택 + fallback 수행
            

            t0 = time.time()
            total = len(idx)

            for k, i in enumerate(idx, 1):
                if (k == 1) or (k % 10 == 0) or (k == total):
                    elapsed = time.time() - t0
                    avg = elapsed / max(1, k)
                    eta = avg * (total - k)
                    print(f"  - LLM deciding {k}/{total} (elapsed {elapsed:.1f}s, ETA ~{eta:.1f}s)", flush=True)

                df.at[i, 'recommended_action'] = self.recommend_action_hybrid(df.loc[i])


        # segment_name
        df['segment_name'] = df.apply(
            lambda row: f"{row['intent']}_{row['urgency']}",
            axis=1
        )

        # Sort by priority (LLM 적용 때문에 이미 정렬되어 있을 수 있지만 다시 보장)
        df = df.sort_values('priority_score', ascending=False).reset_index(drop=True)

        # Apply campaign constraints
        if self.config.max_campaign_users is not None:
            df = df.head(self.config.max_campaign_users)

        print(f"✓ Segmentation complete: {len(df):,} users")
        print(f"  Top priority score: {df['priority_score'].max():.2f}")
        print(f"  Median priority score: {df['priority_score'].median():.2f}")
        print()

        self.segments = df
        return df

    def generate_action_plan(self) -> Dict[str, Any]:
        """
        Generate actionable marketing plan
        
        Returns structured plan with:
        - summary: Overall statistics
        - actions: Action-specific recommendations
        - top_priorities: Highest priority users
        """
        
        print("=" * 80)
        print("GENERATING ACTION PLAN")
        print("=" * 80)
        
        df = self.segments
        
        # Group by recommended action
        action_groups = df.groupby('recommended_action')
        
        actions = []
        
        for action_name, group in action_groups:
            action_info = {
                'action': action_name,
                'count': len(group),
                'avg_probability': round(group['prob'].mean(), 3),
                'avg_eta_days': round(group['eta_days'].mean(), 2) 
                    if 'eta_days' in group.columns else None,
                'avg_priority_score': round(group['priority_score'].mean(), 2),
                'users': group[['user_id', 'category_id', 'prob', 'eta_days', 
                               'priority_score']].to_dict('records')[:10],
            }
            
            # Add action-specific recommendations
            if action_name == MarketingAction.IMMEDIATE_DISCOUNT.value:
                action_info.update({
                    'recommended_timing': 'within 24 hours',
                    'channel': 'email + push notification',
                    'message_tone': 'urgent, limited-time offer',
                    'discount_suggestion': '15-20%',
                    'expected_conversion_lift': '30-50%'
                })
            
            elif action_name == MarketingAction.REMINDER_EMAIL.value:
                action_info.update({
                    'recommended_timing': 'within 2-3 days',
                    'channel': 'email',
                    'message_tone': 'friendly reminder, personalized',
                    'include_elements': ['product images', 'customer reviews', 
                                        'free shipping'],
                    'expected_conversion_lift': '20-35%'
                })
            
            elif action_name == MarketingAction.RETARGETING_AD.value:
                action_info.update({
                    'recommended_timing': 'start immediately, run for 7 days',
                    'channel': 'display ads + social media',
                    'message_tone': 'attention-grabbing, benefit-focused',
                    'creative_type': 'dynamic product ads',
                    'expected_conversion_lift': '15-25%'
                })
            
            elif action_name == MarketingAction.NURTURE_CAMPAIGN.value:
                action_info.update({
                    'recommended_timing': 'multi-touch over 14 days',
                    'channel': 'email drip campaign',
                    'message_tone': 'educational, value-building',
                    'content_type': 'how-to guides, case studies, testimonials',
                    'expected_conversion_lift': '10-20%'
                })
            
            actions.append(action_info)
        
        # Sort by priority
        actions = sorted(actions, key=lambda x: x['avg_priority_score'], reverse=True)
        
        # Top 20 priority users
        top_priorities = df.head(20)[
            ['user_id', 'category_id', 'prob', 'eta_days', 
             'priority_score', 'recommended_action', 'segment_name']
        ].to_dict('records')
        
        # Summary statistics
        summary = {
            'total_users': len(df),
            'total_actions': len(actions),
            'avg_purchase_probability': round(df['prob'].mean(), 3),
            'avg_eta_days': round(df['eta_days'].mean(), 2) 
                if 'eta_days' in df.columns else None,
            'high_priority_users': len(df[df['priority_score'] >= 70]),
            'medium_priority_users': len(df[(df['priority_score'] >= 40) & 
                                           (df['priority_score'] < 70)]),
            'low_priority_users': len(df[df['priority_score'] < 40]),
        }
        
        action_plan = {
            'generated_at': datetime.now().isoformat(),
            'summary': summary,
            'actions': actions,
            'top_priorities': top_priorities,
        }
        
        self.actions = action_plan
        
        print(f"✓ Action plan generated")
        print(f"  Total actions: {len(actions)}")
        print(f"  High priority users: {summary['high_priority_users']:,}")
        print()
        
        return action_plan
    
    def save_outputs(self):
        """Save all outputs to files"""
        
        print("=" * 80)
        print("SAVING OUTPUTS")
        print("=" * 80)
        
        # Save segments CSV
        if self.segments is not None:
            self.segments.to_csv(
                self.config.output_segments_path, 
                index=False, 
                encoding='utf-8-sig'
            )
            print(f"✓ Segments saved: {self.config.output_segments_path}")
        
        # Save action plan JSON
        if self.actions is not None:
            with open(self.config.output_actions_path, 'w', encoding='utf-8') as f:
                json.dump(self.actions, f, indent=2, ensure_ascii=False)
            print(f"✓ Actions saved: {self.config.output_actions_path}")
        
        # Save text report
        self._generate_text_report()
        print(f"✓ Report saved: {self.config.output_report_path}")
        print()
    
    def _generate_text_report(self):
        """Generate human-readable marketing report"""
        
        with open(self.config.output_report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("MARKETING INTELLIGENCE REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 80 + "\n")
            summary = self.actions['summary']
            f.write(f"Total Users Analyzed: {summary['total_users']:,}\n")
            f.write(f"Average Purchase Probability: {summary['avg_purchase_probability']:.1%}\n")
            if summary['avg_eta_days']:
                f.write(f"Average Time to Purchase: {summary['avg_eta_days']:.1f} days\n")
            f.write(f"\nPriority Distribution:\n")
            f.write(f"  High Priority (70+): {summary['high_priority_users']:,} users\n")
            f.write(f"  Medium Priority (40-70): {summary['medium_priority_users']:,} users\n")
            f.write(f"  Low Priority (<40): {summary['low_priority_users']:,} users\n")
            f.write("\n\n")
            
            # Action Breakdown
            f.write("RECOMMENDED ACTIONS\n")
            f.write("-" * 80 + "\n")
            for i, action in enumerate(self.actions['actions'], 1):
                f.write(f"\n{i}. {action['action'].upper().replace('_', ' ')}\n")
                f.write(f"   Users: {action['count']:,}\n")
                f.write(f"   Avg Probability: {action['avg_probability']:.1%}\n")
                if action['avg_eta_days']:
                    f.write(f"   Avg ETA: {action['avg_eta_days']:.1f} days\n")
                f.write(f"   Priority Score: {action['avg_priority_score']:.1f}\n")
                
                if 'recommended_timing' in action:
                    f.write(f"   Timing: {action['recommended_timing']}\n")
                if 'channel' in action:
                    f.write(f"   Channel: {action['channel']}\n")
                if 'expected_conversion_lift' in action:
                    f.write(f"   Expected Lift: {action['expected_conversion_lift']}\n")
            
            f.write("\n\n")
            
            # Top Priorities
            f.write("TOP 10 PRIORITY USERS\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Rank':<6}{'User ID':<20}{'Category ID':<20}")
            f.write(f"{'Prob':<8}{'ETA':<8}{'Score':<8}{'Action':<20}\n")
            f.write("-" * 80 + "\n")
            
            for i, user in enumerate(self.actions['top_priorities'][:10], 1):
                eta_str = f"{user['eta_days']:.1f}d" if not pd.isna(
                    user.get('eta_days')) else "N/A"
                f.write(f"{i:<6}{str(user['user_id']):<20}")
                f.write(f"{str(user['category_id']):<20}")
                f.write(f"{user['prob']:.2f}  {eta_str:<8}")
                f.write(f"{user['priority_score']:<8.1f}")
                f.write(f"{user['recommended_action'][:18]:<20}\n")
            
            f.write("\n" + "=" * 80 + "\n")
    
    def run_pipeline(self):
        """Execute complete agent pipeline"""
        
        print("\n")
        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 78 + "║")
        print("║" + "  MARKETING AGENT LAYER - INTELLIGENCE PIPELINE".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("╚" + "=" * 78 + "╝")
        print("\n")
        
        # Step 1: Load and merge
        self.load_and_merge_predictions()
        
        # Step 2: Segment users
        self.segment_users()
        
        # Step 3: Generate action plan
        self.generate_action_plan()
        
        # Step 4: Save outputs
        self.save_outputs()
        
        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 78 + "║")
        print("║" + "  PIPELINE COMPLETE ✓".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("╚" + "=" * 78 + "╝")
        print("\n")
        
        return self.segments, self.actions


# ============================================================================
# LLM Integration 
# ============================================================================
class LLMMarketingAssistant:
    """
    Solar 필수: 키 없으면 즉시 에러
    모델명은 env로 주입 가능: UPSTAGE_SOLAR_MODEL (기본 solar-mini)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: str = "https://api.upstage.ai/v1",
        poll_interval_sec: float = 2.0,
        timeout_sec: float = 120.0,
        use_file_input: bool = False,  
    ):
        self.api_key = api_key or os.getenv("UPSTAGE_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("UPSTAGE_API_KEY is required (Solar is mandatory).")

        self.model = model or os.getenv("UPSTAGE_SOLAR_MODEL", "solar-pro3")

        self.base_url = base_url
        self.poll_interval_sec = float(poll_interval_sec)
        self.timeout_sec = float(timeout_sec)
        self.use_file_input = bool(use_file_input)

        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
       
        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=20.0,      # <- 20초 넘으면 타임아웃
            max_retries=0      # <- 우리가 아래에서 직접 retry 할 거라 0 추천
        )
        return self._client

    def _solar_run_json(self, payload: Dict, instruction: str) -> Dict:
      
        from openai import APITimeoutError, APIConnectionError, RateLimitError

        client = self._get_client()

        doc = {
            "instruction": instruction,
            "payload": payload,
            "constraints": ["Return ONLY valid JSON.", "No code fences.", "No extra commentary."]
        }

        max_attempts = 3
        backoff_sec = 2.0
        last_err = None

        for attempt in range(1, max_attempts + 1):
            try:
                stream = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "Return ONLY valid JSON. No extra text."},
                        {"role": "user", "content": json.dumps(doc, ensure_ascii=False)}
                    ],
                    stream=True,
                    # ✅ 일단 reasoning 끄기/낮추기 (속도 확인용)
                    reasoning_effort=os.getenv("UPSTAGE_REASONING_EFFORT", "low"),
                )

                chunks = []
                t0 = time.time()
                got_any = False

                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta and getattr(delta, "content", None):
                        if not got_any:
                            got_any = True
                            print(f"\n[LLM] first token after {time.time()-t0:.1f}s", flush=True)
                        chunks.append(delta.content)

                text = "".join(chunks).strip()

                # JSON slicing
                if not text.startswith("{"):
                    s = text.find("{")
                    e = text.rfind("}")
                    if s != -1 and e != -1 and e > s:
                        text = text[s:e+1]

                return json.loads(text)

            except (APITimeoutError, APIConnectionError, RateLimitError) as e:
                last_err = e
                sleep_s = backoff_sec * (2 ** (attempt - 1))
                print(f"[LLM] retry {attempt}/{max_attempts} after error: {e} (sleep {sleep_s:.1f}s)", flush=True)
                time.sleep(sleep_s)
                continue
            except Exception as e:
                last_err = e
                sleep_s = backoff_sec * (2 ** (attempt - 1))
                print(f"[LLM] unexpected error retry {attempt}/{max_attempts}: {e} (sleep {sleep_s:.1f}s)", flush=True)
                time.sleep(sleep_s)
                continue

        raise RuntimeError(f"Solar call failed after {max_attempts} attempts: {last_err}")

    def generate_campaign_copy(self, action_info: Dict) -> str:
       
        safe_action_info = {
            "action": action_info.get("action"),
            "channel": action_info.get("channel"),
            "message_tone": action_info.get("message_tone"),
            "discount_suggestion": action_info.get("discount_suggestion"),
            # 아래는 있으면 쓰고 없으면 절대 언급 금지
            "deadline_iso": action_info.get("deadline_iso"),   # 예: "2026-02-01T23:59:00+09:00"
            "audience_limit": action_info.get("audience_limit") # 예: 5000
        }


        instruction = (
            "You are a marketing copy assistant for a marketer.\n"
            "Generate Korean copies for EMAIL / PUSH / AD.\n"
            "Do not promise guaranteed outcomes.\n"
            "Return JSON with keys: email{subject,body}, push{title,body}, ad{headline,primary_text}."
        )

        out = self._solar_run_json(payload={"action_info": safe_action_info}, instruction=instruction)

        email = out.get("email", {})
        push = out.get("push", {})
        ad = out.get("ad", {})

        parts = []
        parts.append("[EMAIL]\n" + f"Subject: {email.get('subject','')}\n{email.get('body','')}")
        parts.append("[PUSH]\n" + f"Title: {push.get('title','')}\n{push.get('body','')}")
        parts.append("[AD]\n" + f"Headline: {ad.get('headline','')}\n{ad.get('primary_text','')}")
        return "\n\n".join(parts)
    
    def decide_best_action(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rule 후보 중 최종 액션을 LLM(reasoning)으로 선택.
        Return ONLY valid JSON을 강제한다.
        Output JSON schema:
        {
          "action": "reminder_email",
          "why": "<short rationale>",
          "risk": "<short risk/uncertainty note>",
          "guardrails": ["...", "..."]
        }
        """
        instruction = (
            "You are a marketing copy assistant.\n"
            "Generate Korean copies for EMAIL / PUSH / AD.\n"
            "\n"
            "Hard constraints (must follow):\n"
            "1) DO NOT invent any numbers or specifics not provided in action_info.\n"
            "   - No customer counts (e.g., '3,167명'), no satisfaction rates, no deadlines like '24시간',\n"
            "     no '이번 주 한정' unless a real deadline is given.\n"
            "2) DO NOT use guarantee wording (e.g., '보장', '확정', '100%').\n"
            "3) If personalization fields are needed, use placeholders exactly: {customer_name}, {brand_name}.\n"
            "4) Keep claims factual and generic. Use qualitative urgency only (e.g., '기간 한정', '한정 혜택').\n"
            "5) Do not promise guaranteed outcomes.\n"
            "\n"
            "Return ONLY valid JSON with keys:\n"
            "email{subject,body}, push{title,body}, ad{headline,primary_text}.\n"
        )



        out = self._solar_run_json(payload=payload, instruction=instruction)

        # 최소 검증 (후보 외 액션이면 강제 예외 → 상위에서 fallback)
        candidates = payload.get("candidates", [])
        action = out.get("action")
        if action not in candidates:
            raise ValueError(f"LLM chose invalid action: {action} not in candidates={candidates}")

        return out

    
# ============================================================================
# Utilities
# ============================================================================

def validate_inputs(prob_path: str, time_path: str):
    """Validate input file formats"""
    
    required_prob_cols = ['user_id', 'category_id', 'prob']
    required_time_cols = ['user_id', 'category_id', 'eta_days']
    
    # Check probability predictions
    if os.path.exists(prob_path):
        df_prob = pd.read_csv(prob_path, nrows=1)
        missing = set(required_prob_cols) - set(df_prob.columns)
        if missing:
            raise ValueError(f"Probability file missing columns: {missing}")
    
    # Check time predictions
    if os.path.exists(time_path):
        df_time = pd.read_csv(time_path, nrows=1)
        missing = set(required_time_cols) - set(df_time.columns)
        if missing:
            raise ValueError(f"Time predictions file missing columns: {missing}")


def calculate_expected_revenue(
    segments_df: pd.DataFrame,
    avg_order_value: float,
    campaign_cost_per_user: float
) -> pd.DataFrame:
    """Calculate expected revenue and ROI for each segment"""
    
    df = segments_df.copy()
    
    # Expected conversions
    df['expected_conversions'] = df['prob']
    
    # Expected revenue
    df['expected_revenue'] = df['expected_conversions'] * avg_order_value
    
    # Campaign cost
    df['campaign_cost'] = campaign_cost_per_user
    
    # Expected profit
    df['expected_profit'] = df['expected_revenue'] - df['campaign_cost']
    
    # ROI
    df['expected_roi'] = df['expected_profit'] / df['campaign_cost']
    
    return df


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """
    Example usage of Marketing Agent Layer
    
    Demonstrates:
    1. Configuration
    2. Pipeline execution
    3. Results access
    """
    
    # Configure agent
    config = AgentConfig(
        prob_predictions_path="inference_targets.csv",
        time_predictions_path="service_eta_predictions.csv",
        output_segments_path="marketing_segments.csv",
        output_actions_path="marketing_actions.json",
        output_report_path="marketing_report.txt",
        
        # Adjust thresholds for your business
        high_prob_threshold=0.7,
        medium_prob_threshold=0.4,
        urgent_days_threshold=2.0,
        medium_days_threshold=5.0,
        
        # Optional: limit campaign size
        max_campaign_users=10000,
    )
    
    # Validate inputs
    try:
        validate_inputs(
            config.prob_predictions_path, 
            config.time_predictions_path
        )
    except Exception as e:
        print(f"⚠️  Warning: {e}")
    
    # Initialize and run
    agent = MarketingAgentLayer(config)
    segments_df, action_plan = agent.run_pipeline()
    
    # Optional: Calculate ROI
    if segments_df is not None:
        segments_with_roi = calculate_expected_revenue(
            segments_df,
            avg_order_value=100.0,
            campaign_cost_per_user=2.0
        )
        
        print("ROI Analysis:")
        print(segments_with_roi.groupby('recommended_action')['expected_roi'].mean())
    

    
    return segments_df, action_plan


if __name__ == "__main__":
    main()