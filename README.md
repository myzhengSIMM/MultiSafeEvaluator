# MultiSafeEvaluator
Drug development faces a high attrition rate, mainly due to safety concerns in clinical phases — particularly adverse drug reactions (ADRs) caused by off-target effects. To address this, we propose MultiSafeEvaluator, an innovative multi-dimensional evaluation framework for ADR prediction, integrating drug-off-target affinity and pharmacokinetic (PK) parameters to improve the accuracy of drug safety evaluation.
Core Modules
- PreMOTA: Off-target affinity prediction module built on a pre-training-fine-tuning strategy.
- MotifAttNet: PK parameter prediction module developed with motif-level attention mechanisms.
- HetSia-SafeNet: Heterogeneous network module integrating drug-off-target features, PK features, and a learnable ADR representation layer.
Key Results
Experimental results show that MultiSafeEvaluator outperforms state-of-the-art methods in three key tasks:
    Off-target affinity predictionPK parameter predictionADR classification
The framework also enables ADR risk assessment under varying dose conditions.
Significance
This systematic strategy holds great promise for reducing drug development attrition and promoting the advancement of safer drug candidates.
