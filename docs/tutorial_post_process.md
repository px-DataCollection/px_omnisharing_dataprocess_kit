# The Second Part - Format Conversion and Retargeting

**PX Post-Process** is the second part of the pipeline, and its inputs and outputs are as follows:
| Part | Input | Output |
| ---------- | --------------- | --------------- |
| **PX Post-Process** | (a) DF-1; (b) Outputs of pose estimation | DF-2, DF-2R, DF-3 |

Run this part **after the pose estimation complete**.       
Our retargeting framework supports these hand models: MANO, DexH13, and DexH5.   

## Enter the container
```bash
docker start px_pipeline
docker exec -it px_pipeline bash
```

## Running the script
We provide a script `after_pose_main.sh` to run this part of the pipeline. Please first enter the container.      
### Generating DF-2, DF-2R, and DF-3 files in sequence         
```bash
cd pipeline
bash after_pose_main.sh <input_dir> <output_dir> <model> 

# Support batch processing.
```

#### Parameter description
- **input_dir**: The **input** directory. This path **should end with** `vis_res`.                                  
- **output_dir**：The **output** directory. After the script finishes, the generated files are stored in: 
    - DF-2: `${output_dir}/glove`          
    - DF-2R in `${output_dir}/<model>`         
    - DF-3 in `${output_dir}/lerobot`            
- **model**: The retargeting hand model (must be one of `dexh13`/`dexh5`/`mano`)                      
- **WARNING:** the script will **delete and recreate** the path specified by `output_dir`. Do not store any important files in `output_dir`.      

#### Example
```bash
# episodes located at ~/data/batch_1
bash after_pose_main.sh ~/data/batch_1/vis_res ~/data/batch_1/results dexh13
```

#### Incorrect usage examples
The following examples are **wrong** and should **not** be used:                               
```bash
# DO NOT FOLLOW! 
# This will delete the entire '~/data/batch_1' directory
bash after_pose_main.sh ~/data/batch_1/vis_res ~/data/batch_1/ dexh13 

# Do not use named flags like --input/--output/--model
bash after_pose_main.sh --input ~/data/batch_1/vis_res --output ~/data/batch_1/results --model dexh13

# <model> is case-sensitive
bash after_pose_main.sh ~/data/batch_1/vis_res ~/data/batch_1/results MANO # should be 'mano'
bash after_pose_main.sh ~/data/batch_1/vis_res ~/data/batch_1/results DexH13 # should be 'dexh13'
```

### Verifying the input                
`vis_res` is the standard output directory of the preceding component **PX Pose**, and contains both DF-1 data and 6D pose estimates.             
`after_pose_main.sh` assumes a specific directory layout under `vis_res`. The script will fail if it cannot recognize the expected structure.                
- Bimanual (bracelet) poses are **required**.          
- Object poses are **optional**, but providing them improves retargeting accuracy.                 
Whether to include object poses is determined in the previous stage (**PX Pose**). 

### Multi-threaded processing
One stage of PX Post-Process (HDF5 → LeRobot conversion) supports multi-threading. To adjust the level of parallelism, modify the `--num-workers` argument in the following section of `after_pose_main.sh`:             
```bash
python "$SCRIPT_DIR/convert_lerobot/run_converter.py"\
    --data-dir "$RESULT_DIR/$RET_MODEL" \
    --repo-id "/paxini" \
    --task-name "$TASK_NAME"  \
    --fps 30 \
    --num-workers 3 \
    --config "$SCRIPT_DIR/config.json" \
    --output-dir "$RESULT_DIR/lerobot_sep" \
    --encode-videos-bool 
```
Increase or decrease `--num-workers` based on your available CPU cores and system memory.

### Generating DF-2R and DF-3 from DF-2
If your input dataset is already at the DF-2 stage (e.g., the samples from Hugging Face), you can modify `after_pose_main.sh` to process it directly.       
Add the `--from_glove` flag to the `px_retarget` command:
```bash
###########################
# Parsing and Retargeting
###########################
source .venv_post_1/bin/activate
px_retarget --i $SOURCE_DIR --o $RESULT_DIR --model $RET_MODEL --obj_dir $MESH_PATH --gpu -1 --from_glove
```
With the `--from_glove` flag enabled, the pipeline skips the "Parsing" stage, and proceed to retargeting directly.       

## Data validation
The second part of the pipeline utilizes two modules to validate trajectories and filter out abnormal data. One module follows the generation of DF-2, while the other checks the error during retargeting.

### Module 1: Check the poses and tactile data from gloves
Occlusion and certain physical properties of **objects** (e.g., high transparency) can lead to erroneous object pose estimates, which typically manifest as sudden large displacements or rapid change of orientation. These anomalous poses adversely affect downstream retargeting and model training. In addition, a small fraction of the trajectories may contain incorrect tactile readings due to sensor wear.

To address these issues, we introduce a validation mechanism that jointly inspects the pose and tactile data in DF-2 and removes files with abnormal behaviors.         

### Module 2: Collision checking during retargeting
The program enables collision checking on the retargeted hand poses if the input data includes object poses.               
When the program detects severe finger–object penetration and cannot repair it automatically, it may **discard** the corresponding episode to ensure data quality.          

<img src="/images/part_02.png" width="1000">


### Disable data validation
Given the strict criterion, it is possible that all episodes in a batch cannot pass the inspection (especially in _module 1_). In such an occasion, you will see the following error message in the terminal:
```text
Exception: No valid DF-2 files are found. Retargeting will terminate now.
```

To turn off the inspection module, edit this line in `after_pose_main.sh`
```text
px_retarget --i $SOURCE_DIR --o $RESULT_DIR --model $RET_MODEL --obj_dir $MESH_PATH --gpu -1
```

- Adding `--ignore_tactile_check` disables the tactile filter in _Module 1_, while the flag `--ignore_pose_check` disables the pose filter.           
- The `--force_retarget` flag forces trajectory generation after retargeting, regardless of collision or distance errors detected in _Module 2_.
