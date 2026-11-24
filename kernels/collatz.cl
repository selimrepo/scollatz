// Range scanner kernel: one work-item evaluates one start value.
__kernel void collatz_batch(const ulong first, const ulong count,
    __global ulong* steps_out, __global ulong* peak_out, __global uint* status_out,
    const ulong max_steps) {
    ulong gid=(ulong)get_global_id(0); if(gid>=count)return;
    ulong n=first+gid,peak=n,steps=0;uint status=0;
    while(n!=1&&steps<max_steps){if((n&1UL)==0UL)n>>=1;else{if(n>(0xffffffffffffffffUL-1UL)/3UL){status=1;break;}n=n*3UL+1UL;}++steps;if(n>peak)peak=n;}
    if(status==0&&n!=1)status=2;
    steps_out[gid]=steps;peak_out[gid]=peak;status_out[gid]=status;
}

__kernel void collatz_candidates(__global const ulong* candidates, const ulong count,
    __global ulong* steps_out, __global ulong* peak_out, __global uint* status_out,
    const ulong max_steps) {
    ulong gid=(ulong)get_global_id(0); if(gid>=count)return;
    ulong n=candidates[gid],peak=n,steps=0; uint status=0;
    while(n!=1&&steps<max_steps){if((n&1UL)==0UL)n>>=1;else{if(n>(0xffffffffffffffffUL-1UL)/3UL){status=1;break;}n=n*3UL+1UL;}++steps;if(n>peak)peak=n;}
    if(status==0&&n!=1)status=2;
    steps_out[gid]=steps;peak_out[gid]=peak;status_out[gid]=status;
}