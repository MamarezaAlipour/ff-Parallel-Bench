## 64-bit Prime Field Arithmetic Benchmark on GPU with CUDA Backend

```bash
$ DEVICE=gpu make cuda && ./run
```

```bash
Benchmark running on Tesla V100-SXM2-16GB

Addition on F(2^64 - 2^32 + 1)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		        1167380 ns		      0.0695813 ns		           1.43717e+10
256  x  256		    1024		         242704 ns		     0.00361657 ns		           2.76505e+11
512  x  512		    1024		         865201 ns		     0.00322312 ns		           3.10258e+11
1024 x 1024		    1024		        3113482 ns		     0.00289966 ns		           3.44868e+11

Subtraction on F(2^64 - 2^32 + 1)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		          83588 ns		     0.00498223 ns		           2.00713e+11
256  x  256		    1024		         168427 ns		     0.00250976 ns		           3.98445e+11
512  x  512		    1024		         544356 ns		     0.00202788 ns		           4.93125e+11
1024 x 1024		    1024		        2073199 ns		     0.00193082 ns		           5.17915e+11

Multiplication on F(2^64 - 2^32 + 1)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		         221931 ns		      0.0132281 ns		           7.55965e+10
256  x  256		    1024		         547314 ns		     0.00815561 ns		           1.22615e+11
512  x  512		    1024		        1981318 ns		     0.00738098 ns		           1.35483e+11
1024 x 1024		    1024		        7692368 ns		     0.00716408 ns		           1.39585e+11

Division on F(2^64 - 2^32 + 1)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		       19483859 ns		        1.16133 ns		           8.61083e+08
256  x  256		    1024		       46107805 ns		        0.68706 ns		           1.45548e+09
512  x  512		    1024		      190635668 ns		       0.710173 ns		           1.40811e+09
1024 x 1024		    1024		      621002342 ns		       0.578353 ns		           1.72905e+09

Inversion on F(2^64 - 2^32 + 1)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		       19751300 ns		        1.17727 ns		           8.49423e+08
256  x  256		    1024		       41232154 ns		       0.614407 ns		           1.62759e+09
512  x  512		    1024		      161123989 ns		       0.600234 ns		           1.66602e+09
1024 x 1024		    1024		      619209206 ns		       0.576684 ns		           1.73405e+09

Exponentiation on F(2^64 - 2^32 + 1)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		       12272234 ns		       0.731482 ns		           1.36709e+09
256  x  256		    1024		       31862982 ns		       0.474795 ns		           2.10617e+09
512  x  512		    1024		      134662368 ns		       0.501656 ns		            1.9934e+09
1024 x 1024		    1024		      476047384 ns		       0.443354 ns		           2.25554e+09
```
