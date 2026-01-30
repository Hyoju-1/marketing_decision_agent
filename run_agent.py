# -*- coding: utf-8 -*-
"""
Run Marketing Agent Layer 
"""
import os
from agent_layer import (
    AgentConfig,
    MarketingAgentLayer,
    validate_inputs,
    LLMMarketingAssistant,
)

def main():
    if not os.getenv("UPSTAGE_API_KEY"):
        raise RuntimeError("UPSTAGE_API_KEY is required. This project requires Solar integration.")

    # (선택) 모델 지정: export UPSTAGE_SOLAR_MODEL="solar-mini"
    llm = LLMMarketingAssistant()

    config = AgentConfig(
        prob_predictions_path="inference_targets.csv",
        time_predictions_path="service_eta_predictions.csv",
        output_segments_path="marketing_segments.csv",
        output_actions_path="marketing_actions.json",
        output_report_path="marketing_report.txt",
        max_campaign_users=10000,
    )

    try:
        validate_inputs(config.prob_predictions_path, config.time_predictions_path)
    except Exception as e:
        print(f"⚠️ 입력 데이터 검증 경고: {e}")

    agent = MarketingAgentLayer(config, llm_assistant=llm)

    segments_df, action_plan = agent.run_pipeline()

    print("\n실행 완료")
    print(f"총 사용자 수: {len(segments_df):,}")
    print(f"추천 액션 수: {len(action_plan['actions'])}")

    top_action = action_plan["actions"][0]
    print("\n[SOLAR GENERATED COPY]")
    print(llm.generate_campaign_copy(top_action))

if __name__ == "__main__":
    main()

