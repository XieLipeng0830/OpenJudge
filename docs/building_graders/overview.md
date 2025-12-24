# Building Custom Graders

Extend RM-Gallery beyond built-in evaluators by creating custom graders or training reward models. Build domain-specific evaluation logic that seamlessly integrates with RM-Gallery's evaluation pipeline.


## Why Build Custom Graders?

While RM-Gallery provides 50+ pre-built graders, custom graders enable you to evaluate industry-specific criteria (legal, medical, financial), implement proprietary scoring logic, and train models that learn from your preference data. They also help optimize costs by replacing expensive API judges with self-hosted models while maintaining consistent evaluation standards across applications.


## Building Approaches

RM-Gallery supports three paths for creating custom graders. **Create Custom Graders** takes minutes and works best for quick prototyping with code-based logic or LLM-as-judge patterns. **Generate Graders from Data** takes a few hours and automatically builds rubrics from your evaluation examples, ideal for iterative refinement. **Train Reward Models** requires hours to days but delivers the highest scalability for high-volume evaluation with significantly lower inference costs after the initial training investment.

### Decision Framework

```
                         START
                           │
                           ▼
               ┌─────────────────────┐
               │ Have evaluation     │
               │ data with labels?   │
               └──────┬───────┬──────┘
                      │       │
                  YES │       │ NO
                      │       │
                      ▼       ▼
           ┌──────────────┐  ┌──────────────────┐
           │ Want to      │  │ Need evaluation  │
           │ train model? │  │ now?             │
           └────┬────┬────┘  └────┬─────────┬───┘
                │    │            │         │
            YES │    │ NO     YES │         │ NO
                │    │            │         │
                ▼    ▼            ▼         ▼
          ┌──────┐ ┌──────────┐ ┌────────┐ ┌──────────┐
          │Train │ │Generator │ │Custom  │ │ Define   │
          │Model │ │ (Rubric) │ │Graders │ │ criteria │
          └──────┘ └──────────┘ └────────┘ └──────────┘
                │         │           │            │
                └─────────┴───────────┴────────────┘
                              │
                              ▼
                ┌───────────────────────────────┐
                │  Use in evaluation pipeline   │
                │  (GradingRunner, batch eval)  │
                └───────────────────────────────┘
```

If you have labeled evaluation data and want automated learning, train a reward model. If you have data but need faster iteration, use the rubric generator. Without data, create custom graders using code-based logic or LLM judges. Custom graders require no data and setup in under an hour. Generated graders need 50-500 examples and take 1-4 hours. Trained models require 1K-100K examples and 1-3 days but offer 10x lower per-query costs and deterministic consistency, though at the expense of flexibility—changing evaluation criteria requires retraining rather than simple prompt edits.


## Approach 1: Create Custom Graders

Define evaluation logic using LLM judges or code-based functions with no training required. LLM-based graders use models like `qwen3-32b` with custom prompts to evaluate domain-specific criteria (medical accuracy, legal compliance, brand voice). Code-based graders implement deterministic logic—checking response length, keyword presence, format validation, or compliance requirements. Both approaches integrate seamlessly with RM-Gallery's evaluation pipeline.

```python
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.models import OpenAIChatModel

model = OpenAIChatModel(model="qwen3-32b")
grader = LLMGrader(
    name="domain_expert",
    model=model,
    template="Evaluate accuracy: {query} | {response}\nReturn JSON: {{\"score\": <0-1>, \"reason\": \"...\"}}"
)
```

See **[Create Custom Graders →](create_custom_graders.md)** for complete implementation patterns or explore **[Built-in Graders →](../built_in_graders/overview.md)** to customize existing evaluators.


## Approach 2: Generate Graders from Data

GraderGenerator automatically analyzes your evaluation data to create structured scoring rubrics. Provide labeled examples (query-response pairs with scores), and the generator extracts patterns to build interpretable evaluation criteria. This approach works best when you have 50-500 labeled examples but want faster iteration than model training. The generated graders produce explicit rubrics that explain scoring decisions, making them ideal for scenarios requiring transparency and rapid refinement.

```python
from rm_gallery.core.generator import GraderGenerator

generator = GraderGenerator(model=OpenAIChatModel(model="qwen3-32b"))
grader = await generator.generate(
    eval_cases=[{"query": "Q1", "response": "A1", "score": 0.8}, ...],
    task_description="Evaluate response helpfulness"
)
```

See **[Generate Graders from Data →](generate_graders_from_data.md)** for complete workflow and examples.


## Approach 3: Train Reward Models

Train neural networks on preference data to learn evaluation criteria automatically. RM-Gallery supports **Bradley-Terry** training (preference pairs like "Response A > B"), **Generative Pointwise** (absolute scores like "4/5 quality"), and **Generative Pairwise** (comparison decisions). The training workflow involves preparing preference data from human annotations or existing grader outputs, training with VERL's distributed framework (multi-GPU/multi-node support), and deploying the trained model for self-hosted inference. While training requires initial investment (hours to days and sufficient data), the result is highly consistent evaluation at 10x lower per-query cost—ideal for high-volume scenarios exceeding 1M queries per month.

```bash
# Prepare data, train, and integrate
python -m rm_gallery.core.generator.export --dataset helpsteer2 --output-dir ./data
cd tutorials/cookbooks/training_reward_model/bradley_terry && bash run_bt.sh

# Load trained model as grader
model = OpenAIChatModel(model="./checkpoints/my-reward-model", is_local=True)
grader = RelevanceGrader(model=model)
```

Explore **[Training Overview →](training/overview.md)** to compare methods, or start with **[Bradley-Terry Training →](training/bradley_terry.md)** for the most common approach.


## Integration with RM-Gallery

All three approaches produce graders that work identically in RM-Gallery's evaluation pipeline. Call `grader.aevaluate()` for single evaluations or use `GradingRunner` for batch processing. You can combine built-in graders, custom LLM judges, and trained models in a single runner to evaluate responses from multiple perspectives simultaneously.

```python
from rm_gallery.core.runner import GradingRunner

runner = GradingRunner(graders=[RelevanceGrader(), custom_grader, trained_model])
results = await runner.arun_batch([{"query": "Q1", "response": "A1"}, ...])
```


## Tips for Success

**Custom Graders**: Start with code-based logic before adding LLM judges. Test thoroughly on diverse inputs and implement error handling for production use. Track prompt versions for reproducibility and monitor API costs by setting usage limits.

**Generated Graders**: Prioritize 50-100 high-quality examples over hundreds of poor ones. Include edge cases and failure modes in your dataset. Regenerate rubrics iteratively as you collect more data, and validate on held-out samples to ensure generalization.

**Trained Models**: Focus on data quality over quantity—clean preference data matters more than volume. Hold out 10-20% for validation sets. Start with smaller models (1B-7B parameters) and run short training iterations to validate your setup. Monitor evaluation consistency over time to detect drift.


## Complete Example: Build an Evaluation Pipeline

Combine multiple graders for comprehensive assessment. Create a code-based grader for deterministic checks (like length validation), an LLM-based grader for domain-specific accuracy, and optionally add a trained model for learned preferences. Package them in a `GradingRunner` to evaluate multiple responses simultaneously.

```python
from rm_gallery.core.graders.function_grader import FunctionGrader
from rm_gallery.core.graders.llm_grader import LLMGrader
from rm_gallery.core.runner import GradingRunner

# Code-based: length check
length_grader = FunctionGrader(func=lambda r: GraderScore("length", 1.0 if 50<=len(r)<=500 else 0.5), name="length")

# LLM-based: domain accuracy
accuracy_grader = LLMGrader(name="accuracy", model=OpenAIChatModel(model="qwen3-32b"), 
                           template="Rate accuracy: {query}|{response}\nJSON: {{\"score\":<0-1>}}")

# Combine and evaluate
runner = GradingRunner(graders=[length_grader, accuracy_grader])
results = await runner.arun_batch([{"query": "Q", "response": "A"}, ...])
```


## Next Steps

Start with **[Create Custom Graders](create_custom_graders.md)** for immediate results using LLM or code-based logic, or explore **[Built-in Graders](../built_in_graders/overview.md)** to customize existing evaluators. If you have labeled data, use **[Generate Graders from Data](generate_graders_from_data.md)** to auto-generate rubrics, or review **[Training Overview](training/overview.md)** and **[Bradley-Terry Training](training/bradley_terry.md)** for scalable model training. Deploy at scale with **[Run Grading Tasks](../running_graders/run_tasks.md)** for batch workflows, apply graders to **[Refine Data Quality](../applications/data_refinement.md)**, or use **[Pairwise Model Evaluation](../applications/select_rank.md)** to compare and rank models.

