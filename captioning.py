#@title ### **3.2.1. BLIP Captioning**
#@markdown BLIP is a pre-training framework for unified vision-language understanding and generation, which achieves state-of-the-art results on a wide range of vision-language tasks. It can be used as a tool for image captioning, for example, `astronaut riding a horse in space`.
import os

def main(finetune_dir):

    os.chdir(finetune_dir)

    beam_search = True #@param {type:'boolean'}
    min_length = 5 #@param {type:"slider", min:0, max:100, step:5.0}
    max_length = 75 #@param {type:"slider", min:0, max:100, step:5.0}

    config = {
        "_train_data_dir"   : train_data_dir,
        "batch_size"        : 8,
        "beam_search"       : beam_search,
        "min_length"        : min_length,
        "max_length"        : max_length,
        "debug"             : True,
        "caption_extension" : ".caption",
        "max_data_loader_n_workers" : 2,
        "recursive"         : True
    }

    args = ""
    for k, v in config.items():
        if k.startswith("_"):
            args += f'"{v}" '
        elif isinstance(v, str):
            args += f'--{k}="{v}" '
        elif isinstance(v, bool) and v:
            args += f"--{k} "
        elif isinstance(v, float) and not isinstance(v, bool):
            args += f"--{k}={v} "
        elif isinstance(v, int) and not isinstance(v, bool):
            args += f"--{k}={v} "

    final_args = f"python make_captions.py {args}" # Kohya scripts

    os.chdir(finetune_dir)
    !{final_args}
    