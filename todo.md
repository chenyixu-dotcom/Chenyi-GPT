# Next Word Bot TODO

The first milestone is a terminal chatbot that answers the latest message, uses earlier turns, and ends its reply cleanly.

## 1. Make the current generator safe to run

- [ ] Save baseline transcripts for a direct question, a two-turn name recall, a correction, and a long input. Record the checkpoint, seed, and decoding settings.
- [ ] In `src/VanillaAgent.py`, decode only the generated suffix and put token probabilities behind a debug flag.
- [ ] Separate `max_new_tokens` from the model's context limit. With 256 positions and a 64-token reply budget, allow up to 192 prompt tokens, including markers. Handle an oversized latest message with a clear CLI response.
- [ ] Resolve vocabulary and checkpoint paths relative to the project or an explicit config so the CLI can launch from the project root.
- [ ] If keeping the CSV baseline, read the `vocab` column as text with `keep_default_na=False` in both loaders. The current parser maps literal `None` to `<UNK>`; preserving the original row indices keeps checkpoint IDs aligned.

## 2. Clean the data and define a held-out set

- [ ] Convert chat examples into JSONL with one conversation per record and a `messages` array containing `role` and `content`.
- [ ] Keep one Twitter variant or deduplicate across both. Check for private data and document each source and its usage terms.
- [ ] Establish reliable conversation boundaries before using the movie and BNC data. The current parser reads the movie corpus as one block and the notebook samples windows from one flattened token stream.
- [ ] Split by conversation after deduplication. Group by movie or source document where IDs are available. Reserve validation and test data before fitting the tokenizer or creating training windows.
- [ ] Add a small, reviewed batch of actual user–assistant examples for direct answers, follow-ups, corrections, and clarifying questions. Expand categories where held-out examples expose a gap. Use clean movie or BNC dialogue as optional language-model pretraining data.

## 3. Replace the word tokenizer

- [ ] Keep the current `vocab.csv` with its checkpoint as the baseline. Token IDs must stay aligned for that checkpoint to load and behave as trained.
- [ ] Measure token frequencies on the cleaned training split. The current regex finds 78,998 distinct ordinary forms across the raw corpora, and 31,101 occur once. The CSV also contains leading-zero forms such as `007` and `0100`.
- [ ] Train a SentencePiece BPE tokenizer on the training split. Start with 8,000 pieces and byte fallback. Reserve `<BOS>`, `<USER>`, `<ASSISTANT>`, `<EOT>`, and `<PAD>`; insert role markers as reserved IDs.
- [ ] Round-trip leading zeros (`0000`, `007`), decimals, dates, names, URLs, contractions, emoji, and literal role-marker strings. Keep meaningful numbers. Remove recurring junk at the source when it is a verified scraping or formatting artifact.
- [ ] If exact number handling matters, compare SentencePiece's `split_digits` setting on the same samples. Check both round trips and token counts on real prompts because digit-by-digit encoding consumes more context for multi-digit numbers. [SentencePiece options](https://google.github.io/sentencepiece/doc/options/)

## 4. Train assistant replies

- [ ] Move training into a repeatable script and share one chat formatter between training and inference. Use `<USER>…<EOT><ASSISTANT>…<EOT>` during training and end inference prompts with `<ASSISTANT>`.
- [ ] Keep shifted next-token cross-entropy. Set labels to `-100` outside the assistant reply and its ending `<EOT>`, including padding. Inspect one batch to verify that the first assistant word and stop marker are supervised.
- [ ] Overfit a few dozen clean chats as a wiring check, then train on the full training split. Save the tokenizer, model config, checkpoint, seed, and split metadata together.
- [ ] Select a checkpoint using validation loss and generated replies. Use the held-out chat set for the final comparison with the baseline.

## 5. Add session history and verify it

- [ ] Store user and assistant messages in a per-session list, render that list on each turn, and trim the oldest complete exchanges to fit the prompt budget.
- [ ] Stop generation at `<EOT>` or the reply budget. Make reset clear the session history.
- [ ] Score about 30 fixed held-out chats for correct answers, recall, corrections, empty replies, repetition, clean stops, context errors, and latency.
- [ ] If general answers are still weak after the chat behavior works, rerun the same tests with an instruction-tuned backend through the same CLI.

## Publish the project to GitHub

The Git root is the parent `deeplearning` repository, and the checked-out branch during review was `music_lab`. The commands below assume you want a new branch in that same repository. Run them from `next_word_bot` after copying this checklist and the provided `.gitignore` into that folder.

- [ ] Confirm `origin` points to the intended GitHub repository. For a standalone bot repository, use a separate clean checkout and copy in the reviewed files.
- [ ] Inspect the whole index with `git diff --cached --name-only` before staging. If it already contains unrelated work, finish or set aside that staged work before using the commit command below.
- [ ] Review notebooks for saved outputs and sensitive examples. Review the staged diff for credentials and private data. The ignore file covers untracked files that match its patterns. [gitignore documentation](https://git-scm.com/docs/gitignore)

```sh
git switch -c next-word-bot
git status --short --untracked-files=all -- .
git add -- .gitignore TODO.md requirements.txt src/main.py src/VanillaAgent.py src/Trans_model.py
# After reviewing or clearing notebook outputs, add it explicitly if wanted.
# git add -- src/train.ipynb
git diff --cached --name-only
git diff --cached
```

- [ ] Check that every staged path belongs to the bot and that the commit contains only the files you intend to publish.

```sh
git commit -m "Add next word bot and chat roadmap"
git fetch origin
git log --stat --oneline HEAD --not --remotes=origin
git push --dry-run origin HEAD:refs/heads/next-word-bot
```

- [ ] Review the outgoing commits and dry-run destination. When both match what you intend to publish, run `git push -u origin HEAD:refs/heads/next-word-bot`.
- [ ] Document how to obtain or recreate datasets, the vocabulary, and the checkpoint. The current CLI needs `src/vocab.csv` and `src/transformer_model.pth` to run. Review corpus terms before redistributing any corpus or generated artifact.

