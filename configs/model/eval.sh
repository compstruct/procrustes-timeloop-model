cd ../../
scons -j4 --accelergy
cd -
# gdb --args ../../build/timeloop-model VGGN-eval.yaml 
../../build/timeloop-model VGGN-eval.yaml > out.log

# export TIMELOOP_USE_SYNC_PE=true
# ../../build/timeloop-model VGGN-eval.yaml > unbalanced_distance_from_perfect_lb.log
# 
# unset TIMELOOP_USE_SYNC_PE
# ../../build/timeloop-model VGGN-eval.yaml > balanced_distance_from_perfect_lb.log

# gdb --args ../../build/timeloop-model overflow-test.cfg 
# ../../build/timeloop-model speedup-test.cfg > speedup-test.log
# ../../build/timeloop-model overflow-test.cfg > overflow-test.log
# ../../build/timeloop-model bestN.cfg > bestN-fixO-LB-index.log
# ../../build/timeloop-model bk_optimal.cfg > bk_optimal.log
# ../../build/timeloop-model superblock_Conv5_2/fwC2.cfg > fwC2.log
# ../../build/timeloop-model superblock/bwK8.cfg > bwK8.log
# gdb --args ../../build/timeloop-model bk_optimal.cfg
