Genoa Market
=============

<!-- ![banner]() -->
<!-- ![badge]() -->
<!-- ![badge]() -->
This is my implementation of the Genoa Market as described in [Agent-based simulation of a financial market](https://arxiv.org/abs/cond-mat/0103600). This models a single asset stock market among many agents who can either put forward buy or sell orders.

Table of Contents
-----------------

-   [Dependencies](#dependencies)
-   [Simulation Rules](#simulation-rules)
-   [SQL login](#sql-login)
-   [SQL Integration](#sql-integration)
-   [Issues and To-Do](#issues-and-to-do)
-   [Authors](#authors)


Dependencies
---------------

This library really only depends on:
- numpy (for calculating)
- scipy (for interpolating)
- sqlalchemy 1.4 (for talking to sql data base)
- networkx (for coordinating social network)

If you want to use sql features with an existing data base, you need to have the appropriate dialect and drivers. In my home lab I have a mariadb server so I would have to run:

```pip install mariadb```

To get the mariab and mariadbconnector.

Simulation Rules
-----

A Market from the GenoaMarket module requires the number of agents, and an initial price for the stock asset. Optional parameters include a value that defines how clusters form, and the probability that a cluster activates. From there a method is called on the market instance to distribute an initial amount of cash and assets. Then a simulation is run on the market following the loop:

1) Agents in the market try to form social clusters that have the potential to coordinate a trading strategy
2) Agents issue buy and sell orders
3) If enabled, sends data to an sql data base
4) The market uses the issued orders to calculate a market price
5) Orders that have a compatible price limit with the market price are executed as successful transactions between agents.
6) Social clusters are deactivated and destroyed
7) Market increments a counter that keeps track of how many turns have passed by.

See MarketSim.ipynb for an example that generates the log-price-returns time series for two markets. One initialized with a uniform distribution of cash and assets, and one where cash and assets are mostly held by a small amount of agents.

![Alt text](pics/log_price_returns.png?raw=true "Title")

SQL Login
-----
When creating all the utility wrappers for sqlalchemy, I found it useful to save login credentials. This is done by having a json file that looks like:
```
{
    "dialect": "mariadb",
    "driver": "mariadbconnector",
    "pass": "good_password",
    "host": "home_lab.xyz",
    "port": "000"
}
```
See `username.json` for an example.



SQL Integration
-----
 This is done by having some constructor arguments when making a GenoaMarket.Market class. the argument `sql_interact:bool` (false by default) determines if the Market is ever going to even call anything from sqlalchemy. The argument `sql_login:str` is where you put the name of the json file storing the sql login info, if you don't provide one, an sqlite database is used. And `db_name:str` controls what data base to send things to (Note it's expected that tables are made with specific names and column names, see queries/GenoaMarketSchema.sql for the expected schema, when using sqlite, the schema is made for you automatically).

so something like:

```Market(...,sql_interact=True)```

Connects the market to an sqlite database with a generic name.

```Market(...,sql_interact=True,sql_login=".../rod.json",db_name="Genoa")``` 

Connects the market to a server specified in `.../rod.json` directory and the Genoa data base.

```Market(...,sql_interact=False,sql_login=".../rod.json",db_name="Genoa")``` 

doesn't do any sql stuff.

Once this is done the market has an associated sqlalchemy engine and metadata, accessed by `Market.engine` and `Market.meta_data`. Running the simulation then logs:
-   agents (along with there innate properties) in an agents table
-   the market price and buy/sell orders made in response to this price in the market state table
- The orders made by each agent every time step in an orders table
- the portfolio (cash and asset) of each agent every time step

Again please note that table and column names are hardcoded, so use a similar schema as the one in queries/GenoaMarketSchema.sql

Issues and To-Do
-----
- Add methods to let user decide how to handle the following:
    - Invalid drawing of negative values when deciding price limits.
    - How to draw random values when no history is present

- Remake the market price calculation, there are some issues when calculating this for small amount of agents.

- Add analysis functionality, currently this module just creates a foundation to run simulations.

Authors
-------

* Rodrigo Rios [:email:](mailto:rodrigoreyrios@gmail.com)
