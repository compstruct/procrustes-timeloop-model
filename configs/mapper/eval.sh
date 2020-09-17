cd ../../
scons -j16 --accelergy # --d
cd -

# echo "Ideal CK balancing"
# TIMELOOP_GLOBAL_SORT=True ../../build/timeloop-mapper CK.yaml ../../scripts/ERT.yaml > CK_GLB.log

# echo "CK cluster balancing"
# gdb --args ../../build/timeloop-mapper CK.yaml ../../scripts/ERT.yaml
# ../../build/timeloop-mapper CK.yaml ../../scripts/ERT.yaml > CK_CLB.log
# 
# echo "Sparten CK balancing with reordering"
# ../../build/timeloop-mapper CK_reorder.yaml ../../scripts/ERT.yaml > CK_CLB_reorder.log
# ../../build/timeloop-mapper CK_reorder.yaml ../../scripts/ERT.yaml 

# export TIMELOOP_GLOBAL_SORT=True
# export TIMELOOP_DETERMINISTIC_MASK=True
# export TIMELOOP_USE_SYNC_PE=True
# TIMELOOP_GLOBAL_SORT=True ../../build/timeloop-mapper VGGN-1024.yaml > VGGN-1024.log
TIMELOOP_GLOBAL_SORT=True ../../build/timeloop-mapper VGGN.yaml > VGGN-acc32.log
# ../../build/timeloop-mapper depthwise.yaml ../../scripts/ERT.yaml > depthwise.log
# gdb --args ../../build/timeloop-mapper depthwise.yaml ERT.yaml
# ../../build/timeloop-mapper CK.yaml > CK.log
# export TIMELOOP_GLOBAL_SORT=True
# ../../build/timeloop-mapper CK.yaml > CK.log
# unset TIMELOOP_GLOBAL_SORT
# unset TIMELOOP_DETERMINISTIC_MASK
# unset TIMELOOP_USE_SYNC_PE
# ../../build/timeloop-mapper PQ.yaml > PQ.log
# ../../build/timeloop-mapper full-N.yaml > full-N.log
# ../../build/timeloop-mapper FC2.yaml > FC2.log
# ../../build/timeloop-mapper densePQ.yaml > densePQ.log
# gdb --args ../../build/timeloop-mapper FC2.yaml

# CK partition
# ../../build/timeloop-mapper VGG.cfg > VGG-32bit.log
# gdb --args ../../build/timeloop-mapper VGG.cfg

# CN partition
# ../../build/timeloop-mapper VGGN.cfg > VGGN-32bit.log
# ../../build/timeloop-mapper VGGN_bk.cfg > VGGN-32bit_bk.log
# gdb --args ../../build/timeloop-mapper VGGN.cfg
# cp timeloop-mapper.map.cfg ../evaluator/superblock/bwK8.cfg
