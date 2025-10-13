#!/bin/bash

source venv/bin/activate

OUTPUT_FILE="training_output.txt"

echo "Starting training at $(date)" | tee $OUTPUT_FILE
echo "This will take 30-60 minutes..." | tee -a $OUTPUT_FILE
echo "" | tee -a $OUTPUT_FILE

python -m training.train 2>&1 | tee -a $OUTPUT_FILE

echo "" | tee -a $OUTPUT_FILE
echo "Training completed at $(date)" | tee -a $OUTPUT_FILE
echo "" | tee -a $OUTPUT_FILE
echo "Running comparison test..." | tee -a $OUTPUT_FILE
echo "" | tee -a $OUTPUT_FILE

python -m tests.integration.test_evaluate_finetuned 2>&1 | tee -a $OUTPUT_FILE

echo "" | tee -a $OUTPUT_FILE
echo "Generating training plots..." | tee -a $OUTPUT_FILE
python -m training.plot_training

echo "" | tee -a $OUTPUT_FILE
echo "All done!" | tee -a $OUTPUT_FILE
echo "Completed at $(date)" | tee -a $OUTPUT_FILE
echo "Full output saved in $OUTPUT_FILE"
