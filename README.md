# Project Idea
- This project is to build a large language model
- This is not a fine-tuned model, but a large language model made from scratch (almost)
- The model will be trained on a large corpus of text (starting with Wikipedia, later adding more)

# Project Structure
- to be honest, this will be a messy project because I'm not sure how to structure this

# Project Goals
- Build a large language model

# Project Todos
- [ ] Add more data
  - [X] Wikipedia
  - [ ] Books
  - [ ] etc
- [ ] Fine Tune it so that it will be good enough to be used for chatbots
- [ ] Build Tokenizer
- i'll add ones later im lazy

# How to Run
- for nixos users, run `nix develop` in the project directory (make sure the flakes are enabled)

## First Install Dependencies
```sh
uv sync # install the deps
```

## Second Download Data
```sh
uv run -m src.data.download_data # download the data
```

## Third Process Data
```sh
uv run -m src.data.process_data # process the data
```
