## 64-bit Prime Field Arithmetic Benchmark on CPU with OpenCL Backend

```bash
$ DEVICE=cpu make aot_cpu && ./a.out
```

```bash
Benchmark running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

Addition on F(2^64 - 2^32 + 1)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		       12433564 ns		       0.741098 ns		           1.34935e+09
256  x  256		    1024		        3366514 ns		       0.050165 ns		           1.99342e+10
512  x  512		    1024		       11676260 ns		      0.0434975 ns		           2.29898e+10

Subtraction on F(2^64 - 2^32 + 1)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		        6258412 ns		        0.37303 ns		           2.68075e+09
256  x  256		    1024		        2739001 ns		      0.0408143 ns		           2.45012e+10
512  x  512		    1024		        8746381 ns		      0.0325828 ns		            3.0691e+10

Multiplication on F(2^64 - 2^32 + 1)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		        8281675 ns		       0.493626 ns		           2.02582e+09
256  x  256		    1024		        8799549 ns		       0.131123 ns		            7.6264e+09
512  x  512		    1024		       32946489 ns		       0.122735 ns		           8.14762e+09

Division on F(2^64 - 2^32 + 1)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		      235928927 ns		        14.0625 ns		           7.11113e+07
256  x  256		    1024		      914115590 ns		        13.6214 ns		            7.3414e+07
512  x  512		    1024		     3645416787 ns		        13.5802 ns		           7.36364e+07

Inversion on F(2^64 - 2^32 + 1)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		     7371741321 ns		         439.39 ns		           2.27588e+06
256  x  256		    1024		    29427671742 ns		        438.506 ns		           2.28047e+06
512  x  512		    1024		   117261923802 ns		        436.835 ns		            2.2892e+06

Exponentiation on F(2^64 - 2^32 + 1)

  dimension		iterations		          total		                  per op		            ops/ sec
128  x  128		    1024		     4020952486 ns		        239.667 ns		           4.17245e+06
256  x  256		    1024		    16266007109 ns		        242.382 ns		           4.12571e+06
512  x  512		    1024		    64078628014 ns		        238.711 ns		           4.18916e+06
```
