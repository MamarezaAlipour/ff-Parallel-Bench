## 254-bit Prime Field Arithmetic Benchmark on CPU with OpenCL Backend

```bash
$ DEVICE=cpu make aot_cpu && ./a.out
```

```bash
Benchmark running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz


Addition on F(21888242871839275222246405745257275088548364400416034343698204186575808495617)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		       18040546 ns		         1.0753 ns		           9.29973e+08
256  x  256		    1024		        9909792 ns		       0.147667 ns		           6.77198e+09
512  x  512		    1024		       37600646 ns		       0.140073 ns		           7.13912e+09

Subtraction on F(21888242871839275222246405745257275088548364400416034343698204186575808495617)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		       10527422 ns		       0.627483 ns		           1.59367e+09
256  x  256		    1024		       14405879 ns		       0.214664 ns		           4.65844e+09
512  x  512		    1024		       55797658 ns		       0.207862 ns		           4.81087e+09

Multiplication on F(21888242871839275222246405745257275088548364400416034343698204186575808495617)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		     2919444061 ns		        174.012 ns		           5.74672e+06
256  x  256		    1024		    11655611708 ns		        173.682 ns		           5.75764e+06
512  x  512		    1024		    46570145543 ns		        173.487 ns		           5.76411e+06

Division on F(21888242871839275222246405745257275088548364400416034343698204186575808495617)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		     7472763329 ns		        445.411 ns		           2.24512e+06
256  x  256		    1024		    29836573999 ns		          444.6 ns		           2.24921e+06

Inversion on F(21888242871839275222246405745257275088548364400416034343698204186575808495617)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		   109133477527 ns		        6504.86 ns		                153731

Exponentiation on F(21888242871839275222246405745257275088548364400416034343698204186575808495617)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		   198507991247 ns		          11832 ns		               84516.6
```
