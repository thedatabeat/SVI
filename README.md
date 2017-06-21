# Arbitrage-free SVI volatility surfaces

In this project, we introduce an alternative and up to our knowledge new SVI parameterization
of the implied volatility smile in such a way as to guarantee the absence of static
arbitrage. We demonstrate the high quality
of typical SVI fits with a numerical example using data from [finance.yahoo.com](http://finance.yahoo.com).
The analysis is inspired by a paper of Jim Gatheral and 
Antoine Jacquier -- cf. [Arbitrage-free SVI volatility surfaces](https://arxiv.org/pdf/1204.0646.pdf).

The main differences to above mentioned paper are:
* We used a parameterization of the hyperbolas which is numerically stable. 
* We avoid calendar spread arbitrage directly by minimizing the cost function with appropriate penalties. In the paper this was avoided by
testing roots of a 4th order polynomial. We could avoid this without any loss of performance.
