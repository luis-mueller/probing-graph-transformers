#!/usr/bin/env bash

# Run this script from the project root dir.

function run_repeats {
    dataset=$1
    cfg_suffix=$2
    # The cmd line cfg overrides that will be passed to the main.py,
    # e.g. 'name_tag test01 gnn.layer_type gcnconv'
    cfg_overrides=$3

    cfg_file="${cfg_dir}/${dataset}-${cfg_suffix}.yaml"
    if [[ ! -f "$cfg_file" ]]; then
        echo "WARNING: Config does not exist: $cfg_file"
        echo "SKIPPING!"
        return 1
    fi

    main="python main.py --cfg ${cfg_file}"
    out_dir="results/${dataset}"  # <-- Set the output dir.
    common_params="out_dir ${out_dir} ${cfg_overrides}"

    echo "Run program: ${main}"
    echo "  output dir: ${out_dir}"

    #>>> Run each repeat as a separate job <<<
#    for SEED in {0..4}; do
#        script="sbatch ${slurm_directive} -J ${cfg_suffix}-${dataset} run/wrapper.sb ${main} --repeat 1 seed ${SEED} ${common_params}"
##        script="sbatch ${slurm_directive} -J ${cfg_suffix}-${dataset} run/wrapper-narval.sb ${main} --repeat 1 seed ${SEED} ${common_params}"
#        echo $script
#        eval $script
#    done

    #>>> Run all repeats in one job <<<
    script="sbatch ${slurm_directive} -J ${cfg_suffix}-${dataset} run/wrapper.sb ${main} --repeat 10 seed 0 ${common_params}"
    echo $script
    eval $script

    #>>> Run in console <<<
#    script="${main} --repeat 5 seed 0 ${common_params}"
#    echo $script
#    eval $script
}


echo "Do you wish to sbatch jobs? Assuming this is the project root dir: `pwd`"
select yn in "Yes" "No"; do
    case $yn in
        Yes ) break;;
        No ) exit;;
    esac
done



################################################################################
##### 1. Run large hyperparameter grid search for all datasets
################################################################################

# Comment-out runs that you don't want to submit.

cfg_dir="configs/GPS"
slurm_directive="--time=00:30:00 --mem=16G --gres=gpu:1 --cpus-per-task=2"
for DATASET in "actor" "webkb-cor" "webkb-tex" "webkb-wis" "wn-squirrel" "wn-chameleon"; do
  for LYR in 2; do
    for HDIM in 32 64 96; do
      for DRP in 0.0 0.2 0.5 0.8; do
        CPARAM="gt.layers ${LYR}  gt.dim_hidden ${HDIM} gnn.dim_inner ${HDIM}  gt.dropout ${DRP} gnn.dropout ${DRP} gt.attn_dropout 0.0 wandb.use False"
        run_repeats ${DATASET} GPS "name_tag GCN+None+noPE.lyr${LYR}-hdim${HDIM}-drp${DRP}  ${CPARAM} gt.layer_type GCN+None dataset.node_encoder True dataset.node_encoder_name LinearNode posenc_LapPE.enable False"
        run_repeats ${DATASET} GPS "name_tag GCN+None+LPE.lyr${LYR}-hdim${HDIM}-drp${DRP}  ${CPARAM} gt.layer_type GCN+None"
        run_repeats ${DATASET} GPS "name_tag GCN+None+RWSE.lyr${LYR}-hdim${HDIM}-drp${DRP}  ${CPARAM} gt.layer_type GCN+None dataset.node_encoder_name RWSE posenc_LapPE.enable False posenc_RWSE.enable True"
        run_repeats ${DATASET} GPS "name_tag GCN+None+DEG.lyr${LYR}-hdim${HDIM}-drp${DRP}  ${CPARAM} gt.layer_type GCN+None dataset.node_encoder_name LinearNode+GraphormerBias posenc_GraphormerBias.enable True posenc_LapPE.enable False"
        for ATTNDRP in 0.0 0.2 0.5; do
          CPARAM="gt.layers ${LYR}  gt.dim_hidden ${HDIM} gnn.dim_inner ${HDIM}  gt.dropout ${DRP} gnn.dropout ${DRP} gt.attn_dropout ${ATTNDRP} wandb.use False"
          run_repeats ${DATASET} GPS "name_tag GCN+Trf+LPE.lyr${LYR}-hdim${HDIM}-drp${DRP}-attndrp${ATTNDRP}  ${CPARAM} gt.layer_type GCN+Transformer"
          run_repeats ${DATASET} GPS "name_tag GCN+Trf+RWSE.lyr${LYR}-hdim${HDIM}-drp${DRP}-attndrp${ATTNDRP}  ${CPARAM} gt.layer_type GCN+Transformer dataset.node_encoder_name RWSE posenc_LapPE.enable False posenc_RWSE.enable True"
          run_repeats ${DATASET} GPS "name_tag GCN+Trf+DEG.lyr${LYR}-hdim${HDIM}-drp${DRP}-attndrp${ATTNDRP}  ${CPARAM} gt.layer_type GCN+Transformer dataset.node_encoder_name LinearNode+GraphormerBias posenc_GraphormerBias.enable True posenc_LapPE.enable False"
          run_repeats ${DATASET} GPS "name_tag None+Trf+LPE.lyr${LYR}-hdim${HDIM}-drp${DRP}-attndrp${ATTNDRP}  ${CPARAM} gt.layer_type None+Transformer"
          run_repeats ${DATASET} GPS "name_tag None+Trf+RWSE.lyr${LYR}-hdim${HDIM}-drp${DRP}-attndrp${ATTNDRP}  ${CPARAM} gt.layer_type None+Transformer dataset.node_encoder_name RWSE posenc_LapPE.enable False posenc_RWSE.enable True"
          run_repeats ${DATASET} GPS "name_tag None+Trf+DEG.lyr${LYR}-hdim${HDIM}-drp${DRP}-attndrp${ATTNDRP}  ${CPARAM} gt.layer_type None+Transformer dataset.node_encoder_name LinearNode+GraphormerBias posenc_GraphormerBias.enable True posenc_LapPE.enable False"
done; done; done; done; done;


cfg_dir="configs/Graphormer"
slurm_directive="--time=01:45:00 --mem=16G --gres=gpu:rtx8000:1 --cpus-per-task=4"
for DATASET in "actor" "webkb-cor" "webkb-tex" "webkb-wis" "wn-squirrel" "wn-chameleon"; do
  for LYR in 2; do
    for HDIM in 32 64 96; do
      for DRP in 0.0 0.2 0.5 0.8; do
        for ATTNDRP in 0.0 0.2 0.5; do
          cfg_dir="configs/Graphormer"
          CPARAM="graphormer.num_layers ${LYR}  graphormer.embed_dim ${HDIM} gnn.dim_inner ${HDIM}  graphormer.dropout ${DRP} graphormer.mlp_dropout ${DRP} gnn.dropout ${DRP} graphormer.attention_dropout ${ATTNDRP} wandb.use False"
          run_repeats ${DATASET} Graphormer "name_tag full.lyr${LYR}-hdim${HDIM}-drp${DRP}-attndrp${ATTNDRP}  ${CPARAM}"
          run_repeats ${DATASET} Graphormer "name_tag degree.lyr${LYR}-hdim${HDIM}-drp${DRP}-attndrp${ATTNDRP}  ${CPARAM} posenc_GraphormerBias.node_degrees_only True"
done; done; done; done; done;




################################################################################
##### 2. Run final test run with selected configurations
#####    (selection done based on validation performance)
################################################################################

SELECTED=(
"actor-GPS-GCN+None+DEG.lyr2-hdim32-drp0.5"
"actor-GPS-GCN+None+LPE.lyr2-hdim32-drp0.5"
"actor-GPS-GCN+None+RWSE.lyr2-hdim32-drp0.0"
"actor-GPS-GCN+None+noPE.lyr2-hdim96-drp0.8"
"actor-GPS-GCN+Trf+DEG.lyr2-hdim96-drp0.8-attndrp0.2"
"actor-GPS-GCN+Trf+LPE.lyr2-hdim96-drp0.8-attndrp0.2"
"actor-GPS-GCN+Trf+RWSE.lyr2-hdim96-drp0.8-attndrp0.2"
"actor-GPS-None+Trf+DEG.lyr2-hdim96-drp0.2-attndrp0.5"
"actor-GPS-None+Trf+LPE.lyr2-hdim64-drp0.8-attndrp0.5"
"actor-GPS-None+Trf+RWSE.lyr2-hdim96-drp0.2-attndrp0.5"
"webkb-cor-GPS-GCN+None+DEG.lyr2-hdim32-drp0.0"
"webkb-cor-GPS-GCN+None+LPE.lyr2-hdim64-drp0.0"
"webkb-cor-GPS-GCN+None+RWSE.lyr2-hdim96-drp0.2"
"webkb-cor-GPS-GCN+None+noPE.lyr2-hdim96-drp0.2"
"webkb-cor-GPS-GCN+Trf+DEG.lyr2-hdim64-drp0.8-attndrp0.0"
"webkb-cor-GPS-GCN+Trf+LPE.lyr2-hdim96-drp0.5-attndrp0.2"
"webkb-cor-GPS-GCN+Trf+RWSE.lyr2-hdim96-drp0.5-attndrp0.0"
"webkb-cor-GPS-None+Trf+DEG.lyr2-hdim96-drp0.8-attndrp0.0"
"webkb-cor-GPS-None+Trf+LPE.lyr2-hdim64-drp0.5-attndrp0.0"
"webkb-cor-GPS-None+Trf+RWSE.lyr2-hdim96-drp0.8-attndrp0.5"
"webkb-tex-GPS-GCN+None+DEG.lyr2-hdim96-drp0.2"
"webkb-tex-GPS-GCN+None+LPE.lyr2-hdim64-drp0.2"
"webkb-tex-GPS-GCN+None+RWSE.lyr2-hdim96-drp0.0"
"webkb-tex-GPS-GCN+None+noPE.lyr2-hdim96-drp0.2"
"webkb-tex-GPS-GCN+Trf+DEG.lyr2-hdim96-drp0.8-attndrp0.0"
"webkb-tex-GPS-GCN+Trf+LPE.lyr2-hdim64-drp0.8-attndrp0.0"
"webkb-tex-GPS-GCN+Trf+RWSE.lyr2-hdim64-drp0.8-attndrp0.2"
"webkb-tex-GPS-None+Trf+DEG.lyr2-hdim96-drp0.8-attndrp0.5"
"webkb-tex-GPS-None+Trf+LPE.lyr2-hdim64-drp0.8-attndrp0.5"
"webkb-tex-GPS-None+Trf+RWSE.lyr2-hdim96-drp0.8-attndrp0.2"
"webkb-wis-GPS-GCN+None+DEG.lyr2-hdim64-drp0.2"
"webkb-wis-GPS-GCN+None+LPE.lyr2-hdim64-drp0.5"
"webkb-wis-GPS-GCN+None+RWSE.lyr2-hdim64-drp0.5"
"webkb-wis-GPS-GCN+None+noPE.lyr2-hdim96-drp0.5"
"webkb-wis-GPS-GCN+Trf+DEG.lyr2-hdim96-drp0.8-attndrp0.0"
"webkb-wis-GPS-GCN+Trf+LPE.lyr2-hdim96-drp0.8-attndrp0.0"
"webkb-wis-GPS-GCN+Trf+RWSE.lyr2-hdim96-drp0.8-attndrp0.0"
"webkb-wis-GPS-None+Trf+DEG.lyr2-hdim96-drp0.8-attndrp0.5"
"webkb-wis-GPS-None+Trf+LPE.lyr2-hdim96-drp0.5-attndrp0.0"
"webkb-wis-GPS-None+Trf+RWSE.lyr2-hdim96-drp0.8-attndrp0.2"
"wn-chameleon-GPS-GCN+None+DEG.lyr2-hdim96-drp0.0"
"wn-chameleon-GPS-GCN+None+LPE.lyr2-hdim32-drp0.5"
"wn-chameleon-GPS-GCN+None+RWSE.lyr2-hdim96-drp0.2"
"wn-chameleon-GPS-GCN+None+noPE.lyr2-hdim96-drp0.0"
"wn-chameleon-GPS-GCN+Trf+DEG.lyr2-hdim96-drp0.0-attndrp0.0"
"wn-chameleon-GPS-GCN+Trf+LPE.lyr2-hdim96-drp0.0-attndrp0.2"
"wn-chameleon-GPS-GCN+Trf+RWSE.lyr2-hdim64-drp0.0-attndrp0.5"
"wn-chameleon-GPS-None+Trf+DEG.lyr2-hdim96-drp0.8-attndrp0.5"
"wn-chameleon-GPS-None+Trf+LPE.lyr2-hdim64-drp0.8-attndrp0.2"
"wn-chameleon-GPS-None+Trf+RWSE.lyr2-hdim64-drp0.5-attndrp0.5"
"wn-squirrel-GPS-GCN+None+DEG.lyr2-hdim96-drp0.0"
"wn-squirrel-GPS-GCN+None+LPE.lyr2-hdim96-drp0.0"
"wn-squirrel-GPS-GCN+None+RWSE.lyr2-hdim64-drp0.0"
"wn-squirrel-GPS-GCN+None+noPE.lyr2-hdim32-drp0.0"
"wn-squirrel-GPS-GCN+Trf+DEG.lyr2-hdim64-drp0.0-attndrp0.5"
"wn-squirrel-GPS-GCN+Trf+LPE.lyr2-hdim96-drp0.0-attndrp0.5"
"wn-squirrel-GPS-GCN+Trf+RWSE.lyr2-hdim64-drp0.0-attndrp0.5"
"wn-squirrel-GPS-None+Trf+DEG.lyr2-hdim32-drp0.8-attndrp0.2"
"wn-squirrel-GPS-None+Trf+LPE.lyr2-hdim32-drp0.5-attndrp0.5"
"wn-squirrel-GPS-None+Trf+RWSE.lyr2-hdim64-drp0.8-attndrp0.0"
)
cfg_dir="configs/GPS"
slurm_directive="--time=00:50:00 --mem=16G --gres=gpu:1 --cpus-per-task=2"
for DATASET in "actor" "webkb-cor" "webkb-tex" "webkb-wis" "wn-squirrel" "wn-chameleon"; do
  for LYR in 2; do
    for HDIM in 32 64 96; do
      for DRP in 0.0 0.2 0.5 0.8; do
        CPARAM="gt.layers ${LYR}  gt.dim_hidden ${HDIM} gnn.dim_inner ${HDIM}  gt.dropout ${DRP} gnn.dropout ${DRP} gt.attn_dropout 0.0  wandb.use False"

        EXPID="${DATASET}-GPS-GCN+None+noPE.lyr${LYR}-hdim${HDIM}-drp${DRP}"
        if printf '%s\0' "${SELECTED[@]}" | grep -Fxqz -- "${EXPID}"; then
            run_repeats ${DATASET} GPS "name_tag GCN+None+noPE.lyr${LYR}-hdim${HDIM}-drp${DRP}  ${CPARAM} gt.layer_type GCN+None dataset.node_encoder True dataset.node_encoder_name LinearNode posenc_LapPE.enable False"
        fi

        EXPID="${DATASET}-GPS-GCN+None+LPE.lyr${LYR}-hdim${HDIM}-drp${DRP}"
        if printf '%s\0' "${SELECTED[@]}" | grep -Fxqz -- "${EXPID}"; then
            run_repeats ${DATASET} GPS "name_tag GCN+None+LPE.lyr${LYR}-hdim${HDIM}-drp${DRP}  ${CPARAM} gt.layer_type GCN+None"
        fi

        EXPID="${DATASET}-GPS-GCN+None+RWSE.lyr${LYR}-hdim${HDIM}-drp${DRP}"
        if printf '%s\0' "${SELECTED[@]}" | grep -Fxqz -- "${EXPID}"; then
            run_repeats ${DATASET} GPS "name_tag GCN+None+RWSE.lyr${LYR}-hdim${HDIM}-drp${DRP}  ${CPARAM} gt.layer_type GCN+None dataset.node_encoder_name RWSE posenc_LapPE.enable False posenc_RWSE.enable True"
        fi

        EXPID="${DATASET}-GPS-GCN+None+DEG.lyr${LYR}-hdim${HDIM}-drp${DRP}"
        if printf '%s\0' "${SELECTED[@]}" | grep -Fxqz -- "${EXPID}"; then
            run_repeats ${DATASET} GPS "name_tag GCN+None+DEG.lyr${LYR}-hdim${HDIM}-drp${DRP}  ${CPARAM} gt.layer_type GCN+None dataset.node_encoder_name LinearNode+GraphormerBias posenc_GraphormerBias.enable True posenc_LapPE.enable False"
        fi

        for ATTNDRP in 0.0 0.2 0.5; do
          CPARAM="gt.layers ${LYR}  gt.dim_hidden ${HDIM} gnn.dim_inner ${HDIM}  gt.dropout ${DRP} gnn.dropout ${DRP} gt.attn_dropout ${ATTNDRP}  wandb.use False"

          EXPID="${DATASET}-GPS-GCN+Trf+LPE.lyr${LYR}-hdim${HDIM}-drp${DRP}-attndrp${ATTNDRP}"
          if printf '%s\0' "${SELECTED[@]}" | grep -Fxqz -- "${EXPID}"; then
              run_repeats ${DATASET} GPS "name_tag GCN+Trf+LPE.lyr${LYR}-hdim${HDIM}-drp${DRP}-attndrp${ATTNDRP}  ${CPARAM} gt.layer_type GCN+Transformer"
          fi

          EXPID="${DATASET}-GPS-GCN+Trf+RWSE.lyr${LYR}-hdim${HDIM}-drp${DRP}-attndrp${ATTNDRP}"
          if printf '%s\0' "${SELECTED[@]}" | grep -Fxqz -- "${EXPID}"; then
              run_repeats ${DATASET} GPS "name_tag GCN+Trf+RWSE.lyr${LYR}-hdim${HDIM}-drp${DRP}-attndrp${ATTNDRP}  ${CPARAM} gt.layer_type GCN+Transformer dataset.node_encoder_name RWSE posenc_LapPE.enable False posenc_RWSE.enable True"
          fi

          EXPID="${DATASET}-GPS-GCN+Trf+DEG.lyr${LYR}-hdim${HDIM}-drp${DRP}-attndrp${ATTNDRP}"
          if printf '%s\0' "${SELECTED[@]}" | grep -Fxqz -- "${EXPID}"; then
              run_repeats ${DATASET} GPS "name_tag GCN+Trf+DEG.lyr${LYR}-hdim${HDIM}-drp${DRP}-attndrp${ATTNDRP}  ${CPARAM} gt.layer_type GCN+Transformer dataset.node_encoder_name LinearNode+GraphormerBias posenc_GraphormerBias.enable True posenc_LapPE.enable False"
          fi

          EXPID="${DATASET}-GPS-None+Trf+LPE.lyr${LYR}-hdim${HDIM}-drp${DRP}-attndrp${ATTNDRP}"
          if printf '%s\0' "${SELECTED[@]}" | grep -Fxqz -- "${EXPID}"; then
              run_repeats ${DATASET} GPS "name_tag None+Trf+LPE.lyr${LYR}-hdim${HDIM}-drp${DRP}-attndrp${ATTNDRP}  ${CPARAM} gt.layer_type None+Transformer"
          fi

          EXPID="${DATASET}-GPS-None+Trf+RWSE.lyr${LYR}-hdim${HDIM}-drp${DRP}-attndrp${ATTNDRP}"
          if printf '%s\0' "${SELECTED[@]}" | grep -Fxqz -- "${EXPID}"; then
              run_repeats ${DATASET} GPS "name_tag None+Trf+RWSE.lyr${LYR}-hdim${HDIM}-drp${DRP}-attndrp${ATTNDRP}  ${CPARAM} gt.layer_type None+Transformer dataset.node_encoder_name RWSE posenc_LapPE.enable False posenc_RWSE.enable True"
          fi

          EXPID="${DATASET}-GPS-None+Trf+DEG.lyr${LYR}-hdim${HDIM}-drp${DRP}-attndrp${ATTNDRP}"
          if printf '%s\0' "${SELECTED[@]}" | grep -Fxqz -- "${EXPID}"; then
              run_repeats ${DATASET} GPS "name_tag None+Trf+DEG.lyr${LYR}-hdim${HDIM}-drp${DRP}-attndrp${ATTNDRP}  ${CPARAM} gt.layer_type None+Transformer dataset.node_encoder_name LinearNode+GraphormerBias posenc_GraphormerBias.enable True posenc_LapPE.enable False"
          fi
done; done; done; done; done;




SELECTED=(
"actor-Graphormer-degree.lyr2-hdim64-drp0.8-attndrp0.0"
"actor-Graphormer-full.lyr2-hdim96-drp0.2-attndrp0.5"
"webkb-cor-Graphormer-degree.lyr2-hdim96-drp0.5-attndrp0.0"
"webkb-cor-Graphormer-full.lyr2-hdim96-drp0.5-attndrp0.0"
"webkb-tex-Graphormer-degree.lyr2-hdim64-drp0.2-attndrp0.5"
"webkb-tex-Graphormer-full.lyr2-hdim64-drp0.0-attndrp0.2"
"webkb-wis-Graphormer-degree.lyr2-hdim96-drp0.2-attndrp0.0"
"webkb-wis-Graphormer-full.lyr2-hdim96-drp0.5-attndrp0.0"
"wn-squirrel-Graphormer-degree.lyr2-hdim96-drp0.0-attndrp0.5"
"wn-squirrel-Graphormer-full.lyr2-hdim96-drp0.0-attndrp0.0"
"wn-chameleon-Graphormer-degree.lyr2-hdim96-drp0.2-attndrp0.0"
"wn-chameleon-Graphormer-full.lyr2-hdim96-drp0.2-attndrp0.0"
)
cfg_dir="configs/Graphormer"
slurm_directive="--time=03:30:00 --mem=16G --gres=gpu:rtx8000:1 --cpus-per-task=4"
for DATASET in "actor" "webkb-cor" "webkb-tex" "webkb-wis" "wn-squirrel" "wn-chameleon"; do
  for LYR in 2; do
    for HDIM in 32 64 96; do
      for DRP in 0.0 0.2 0.5 0.8; do
        for ATTNDRP in 0.0 0.2 0.5; do
          CPARAM="graphormer.num_layers ${LYR}  graphormer.embed_dim ${HDIM} gnn.dim_inner ${HDIM}  graphormer.dropout ${DRP} graphormer.mlp_dropout ${DRP} gnn.dropout ${DRP} graphormer.attention_dropout ${ATTNDRP} wandb.use False"

          EXPID="${DATASET}-Graphormer-full.lyr${LYR}-hdim${HDIM}-drp${DRP}-attndrp${ATTNDRP}"
          if printf '%s\0' "${SELECTED[@]}" | grep -Fxqz -- "${EXPID}"; then
              run_repeats ${DATASET} Graphormer "name_tag full.lyr${LYR}-hdim${HDIM}-drp${DRP}-attndrp${ATTNDRP}  ${CPARAM}"
          fi

          EXPID="${DATASET}-Graphormer-degree.lyr${LYR}-hdim${HDIM}-drp${DRP}-attndrp${ATTNDRP}"
          if printf '%s\0' "${SELECTED[@]}" | grep -Fxqz -- "${EXPID}"; then
              run_repeats ${DATASET} Graphormer "name_tag degree.lyr${LYR}-hdim${HDIM}-drp${DRP}-attndrp${ATTNDRP}  ${CPARAM} posenc_GraphormerBias.node_degrees_only True"
          fi
done; done; done; done; done;
