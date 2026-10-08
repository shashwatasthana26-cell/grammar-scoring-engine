# grammar-scoring-engine
Grammar scoring engine for spoken audio (0-5 scale). Whisper transcribes each clip; nine rubric-based features (grammar errors, sentence length, vocabulary diversity, fillers, pauses, speech rate, ASR confidence) feed a Ridge regression. 5-fold CV: RMSE 0.838 vs 1.238 baseline, Pearson 0.736. Built for the SHL Hiring Assessment 2026.
