#!/bin/bash

# Define hyperparameter ranges
learning_rates=(1e-4 1e-5)      # Learning rates
weight_decays=(1e-4 1e-5 1e-6)  # Weight decay values
batch_sizes=(16 32 64)          # Batch sizes
num_heads=(4 8)              # Number of attention heads
num_layers=(6 8 10)             # Number of transformer layers
projection_dims=(256 512 1024)  # Projection dimensions

# Output directory
OUTPUT_DIR="hyperparam_outputs"
mkdir -p "$OUTPUT_DIR"  # Create the output directory if it doesn't exist

# Record the start time
start_time=$(date)

# Hyperparameter tuning loop
for lr in "${learning_rates[@]}"; do
  for wd in "${weight_decays[@]}"; do
    for bs in "${batch_sizes[@]}"; do
      for nh in "${num_heads[@]}"; do
        for nl in "${num_layers[@]}"; do
          for pd in "${projection_dims[@]}"; do
            # Generate a unique filename for the current hyperparameter combination
            OUTPUT_FILE="${OUTPUT_DIR}/lr_${lr}_wd_${wd}_bs_${bs}_nh_${nh}_nl_${nl}_pd_${pd}.log"
            
            echo "Starting: lr=$lr, wd=$wd, bs=$bs, nh=$nh, nl=$nl, pd=$pd"
            echo "Output will be saved to: $OUTPUT_FILE"
            
            # Run the Python training script and redirect output to the log file
            python3 regression/main.py \
              --hyperparameter true \
              --lr "$lr" \
              --weight_decay "$wd" \
              --batch_size "$bs" \
              --num_heads "$nh" \
              --num_layers "$nl" \
              --projection_dim "$pd" > "$OUTPUT_FILE"
            
            # Log completion status
            echo "Completed: lr=$lr, wd=$wd, bs=$bs, nh=$nh, nl=$nl, pd=$pd. Output saved to $OUTPUT_FILE."
          done
        done
      done
    done
  done
done

# Record the end time
end_time=$(date)

# Summary
echo "Hyperparameter tuning completed."
echo "Log files are saved in the $OUTPUT_DIR directory."
echo "Start time: $start_time"
echo "End time: $end_time"
