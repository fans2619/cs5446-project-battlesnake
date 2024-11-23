#!/bin/bash

# Run both scripts in the background
python main.py &
python mcts_main.py &

# Wait for all background processes to finish
wait
