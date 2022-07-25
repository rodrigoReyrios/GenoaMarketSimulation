Genoa Market
=============

<!-- ![banner]() -->
<!-- ![badge]() -->
<!-- ![badge]() -->
This is my implementation of the Genoa Market as described in [Agent-based simulation of a financial market](https://arxiv.org/abs/cond-mat/0103600). This models a single asset stock market among many agents who can either put forward buy or sell orders.

Table of Contents
-----------------

-   [Dependencies](#dependencies)
-   [Simulation Rules](#simulationtrules)
-   [ToDo](#issuesandtodo)
-   [Authors](#authors)


Dependencies
---------------

GenoaMarket module depends on networkx, numpy, and scipy. All other libraries should be included in standard libraries of python 3.

Simulation Rules
-----

A Market from the GenoaMarket module requires the number of agents, and an initial price for the stock asset. Optional parameters include a value that defines how clusters form, and the probability that a cluster activates. From there a method is called on the market instance to distribute an initial amount of cash and assets. Then a simulation is run on the market following the loop:

1) Agents in the market try to form social clusters that have the potential to coordinate a trading strategy
2) Agents issue buy and sell orders
3) The market uses the issued orders to calculate a market price
4) Orders that have a compatible price limit with the market price are executed as successful transactions between agents.
5) Social clusters are deactivated and destroyed
6) Market increments a counter that keeps track of how many turns have passed by.

See MarketSim.ipynb for an example that generates the log-price-returns time series for two markets. One initialized with a uniform distribution of cash and assets, and one where cash and assets are mostly held by a small amount of agents.

![Alt text](pics/log_price_returns.png?raw=true "Title")


Issues and To-Do
-----
- Add methods to let user decide how to handle the following:
    - Invalid drawing of negative values when deciding price limits.
    - How to draw random values when no history is present

- Remake the market price calculation, there are some issues when calculating this for small amount of agents.

- Add functionality to send simulation data to an SQL server.

- Add analysis functionality, currently this module just creates a foundation to run simulations.

Authors
-------

* Rodrigo Rios [:email:](mailto:rodrigoreyrios@gmail.com)
