 #!/bin/bash
 docker run --gpus all -v $(pwd):/workspace -e RB_PATH=/workspace/data/rb_1000_20220727T2221.pkl -it lhdorl