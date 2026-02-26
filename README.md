# character-level-_ransformer_for_text_generation
Comparing 8 sampling methods to get clear, reusable insights into when and how to use each method.
●	Problem: Autoregressive language models rely on decoding strategies to generate text. While sampling methods and hyperparameters (temperature, top-k, top-p) critically control the diversity-coherence trade-off, they lack systematic, controlled comparison—especially on character-level models. Furthermore, advanced methods like Mirostat and Typical Sampling remain underexplored in small-scale, controlled testbeds.
●	Objective: To implement a character-level autoregressive Transformer and conduct a controlled comparative study of eight distinct sampling strategies:
o	Greedy Search (Deterministic baseline)
o	Beam Search (Width = 3, 5)
o	Pure Random Sampling (Ancestral)	
o	Temperature Scaling
o	Top-k Sampling
o	Top-p (Nucleus) Sampling
o	Typical Sampling
o	Mirostat (Adaptive dynamic sampling)
●	Research Question: How do these eight sampling strategies compare in managing the trade-off between output diversity and textual coherence in character-level text generation? Specifically, which methods produce the most "Shakespearean-like" text while avoiding the pitfalls of repetition (greedy methods) or incoherence (high-temperature methods)?
●	Sampling methods strongly affect generative models but lack systematic study. Character-level models offer a simple testbed. This research will give clear, reusable insights into when and how to use each method.
