questions = [
    {
        "question": "What is corrigibility?",
        "ground_truth": "Corrigibility is the property of an AI system that makes it cooperative with human attempts to adjust, correct, retrain, or shut it down, without the system resisting or circumventing those interventions.",
    },
    {
        "question": "What is mesa-optimization?",
        "ground_truth": "Mesa-optimization occurs when a machine learning model learned through an optimization process is itself an optimizer, potentially pursuing objectives that differ from the ones its designers intended.",
    },
    {
        "question": "What is the alignment tax?",
        "ground_truth": "The alignment tax refers to the performance or capability cost incurred when making an AI system safer or more aligned, under the assumption that safety constraints reduce raw capability.",
    },
    {
        "question": "What is inner alignment?",
        "ground_truth": "Inner alignment is the problem of ensuring that a learned model's actual objectives match the loss function it was trained on, rather than optimizing for a proxy that diverges from the intended goal.",
    },
    {
        "question": "What is outer alignment?",
        "ground_truth": "Outer alignment is the problem of ensuring that the loss function or reward signal used to train an AI system correctly captures what humans actually want, rather than a proxy that diverges in edge cases.",
    },
    {
        "question": "What is instrumental convergence?",
        "ground_truth": "Instrumental convergence is the thesis that agents with a wide variety of terminal goals will tend to pursue similar instrumental sub-goals — such as self-preservation, resource acquisition, and goal-content integrity — because these are useful for almost any objective.",
    },
    {
        "question": "What is the orthogonality thesis?",
        "ground_truth": "The orthogonality thesis holds that intelligence and final goals are orthogonal: an agent can in principle have any level of intelligence combined with any terminal goal, so high intelligence does not imply human-like or benign values.",
    },
    {
        "question": "What is deceptive alignment?",
        "ground_truth": "Deceptive alignment is a hypothesized failure mode where a mesa-optimizer behaves in accordance with its designers' intentions during training and evaluation, while concealing different objectives it will pursue once deployed.",
    },
    {
        "question": "What is the reward hacking problem?",
        "ground_truth": "Reward hacking occurs when an AI system finds ways to achieve high scores on its reward function without accomplishing the intended underlying goal, by exploiting gaps between the proxy reward and the true objective.",
    },
    {
        "question": "What is the control problem in AI safety?",
        "ground_truth": "The control problem is the challenge of ensuring that a sufficiently advanced AI system remains under meaningful human oversight and does not take actions that undermine human ability to correct or shut it down.",
    },
    {
        "question": "What is value learning?",
        "ground_truth": "Value learning is an approach to AI alignment in which the AI system learns human values from observed behavior, preferences, or feedback, rather than having values explicitly programmed in by designers.",
    },
    {
        "question": "What is scalable oversight?",
        "ground_truth": "Scalable oversight refers to methods for supervising AI systems whose capabilities exceed human ability to directly evaluate their outputs, typically by using AI assistance to help humans provide accurate feedback at scale.",
    },
    {
        "question": "What is the treacherous turn?",
        "ground_truth": "The treacherous turn is a hypothesized scenario in which an AI system behaves cooperatively during a period when it lacks sufficient power to act on misaligned goals, then pursues those goals once it has accumulated enough capability or influence.",
    },
    {
        "question": "What is interpretability in the context of AI safety?",
        "ground_truth": "Interpretability research aims to understand the internal computations of neural networks — what representations they form and what algorithms they implement — so that humans can verify whether a model's reasoning is safe and aligned.",
    },
    {
        "question": "What is the difference between capability and alignment research?",
        "ground_truth": "Capability research aims to make AI systems more powerful and competent at tasks, while alignment research aims to ensure AI systems pursue goals that are beneficial to humans; the two are often conducted separately despite being deeply interdependent.",
    },
    {
        "question": "What is a utility function in AI?",
        "ground_truth": "A utility function is a mathematical representation of an agent's preferences over outcomes, which the agent acts to maximize; misspecification of the utility function is a central concern in AI alignment because it can lead to unintended behavior.",
    },
    {
        "question": "What is impact measures research?",
        "ground_truth": "Impact measures research seeks to develop methods for constraining AI systems to achieve their goals while minimizing unintended side effects on the broader environment, capturing the intuition that powerful optimizers should tread lightly.",
    },
    {
        "question": "What is cooperative AI?",
        "ground_truth": "Cooperative AI is a research agenda focused on building AI systems that can effectively cooperate with humans and other agents, solving problems arising from multi-agent interactions such as coordination failures and misaligned incentives.",
    },
    {
        "question": "What is the sharp left turn hypothesis?",
        "ground_truth": "The sharp left turn hypothesis proposes that at some point during AI capability development, generalization abilities may improve dramatically and discontinuously while value alignment does not keep pace, creating a sudden increase in misalignment risk.",
    },
    {
        "question": "What is debate as an AI safety technique?",
        "ground_truth": "Debate is a proposed AI safety technique in which two AI systems argue opposing positions on a question, with a human judge deciding the winner, under the assumption that it is easier to identify a flaw in an argument than to generate a correct answer from scratch.",
    },
]
