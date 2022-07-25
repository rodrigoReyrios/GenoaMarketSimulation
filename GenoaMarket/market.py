"""Module Containing Market class that controls many agents."""

from dataclasses import dataclass, field
from math import floor
from time import perf_counter
import numpy as np
import networkx as nx
from .agent import Agent
from .misc import make_time_string, make_demand_supply, DType


@dataclass
class Market:
    """
    Class Representing a Market of many agents.

    Attributes
    ----------
    N :int
        The number of agents in this market
    init_price :float
        The initial price of the asset in this Market
    pa :float = 0.01
        Float determining the probability that any two agents develope a edge in the
        social network
    pc :float = 0.2
        Probability that a cluster (connected component of the social graph) is activated
        and all agents in a random cluster either sell or buy based on a 50/50 coin flip
    age :int = 0
        The age in 'turns' of the market
    prices :list[float]
        The historical prices of the asset in the market
    agents :list[Agent]
        A list of the actual agent instances populating the market
    network :networkx.classes.graph.Graph
        A undirected graph representing the social network of the agents
    """

    N: int
    init_price: float = field(repr=False)
    pa: float = field(default=0.01)
    pc: float = field(default=0.2)
    age: int = field(default=0, init=False)
    prices: list[float] = field(init=False, repr=False)
    agents: list[Agent] = field(init=False, repr=False)
    network: nx.classes.graph.Graph = field(init=False, repr=False)

    def __post_init__(self):
        # populate the agents array with self.N agents
        self.agents = [
            Agent(C=0.0, A=0.0, Ti=20, prob=0.5, ind=i) for i in range(self.N)
        ]
        # start keeping track of the prices
        self.prices = [self.init_price]
        # initialize a network and add a node per agent
        self.network = nx.Graph()
        self.network.add_nodes_from(range(self.N))

    def make_network(self):
        """
        Adds random edges based on self.pa to the network to build the social network.

        Args:
            N/A
        Returns:
            N/A
        Raises:
            N/A
        """
        # draw the probabilities per edge
        edge_prob = np.random.uniform(size=int(self.N * (self.N + 1) / 2))
        # start a counter to keep track of which edge were looking at
        im = 0
        # build edge list (i,j)
        E = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                # check if this edge passes
                if edge_prob[im] <= self.pa:
                    E.append((i, j))
                im += 1
        self.network.add_edges_from(E)

    def reset_network(self):
        """
        Resets the social network to remove any existing edges.

        Args:
            N/A
        Returns:
            N/A
        Raises:
            N/A
        """
        # remove all the edges from the network
        self.network.remove_edges_from(self.network.edges)

    def distribute(self, C, A, dtype=DType.UNIFORM):
        """
        This method distributes an amount of cash and assets among the markets agents.

        Args:
            C :float - The amount of cash to distribute
            A :int - The amount of asset to distribute to the agents
            dtype :Dtype = DType.UNIFORM - The method used to distribute the cash and assets
        Returns:
            N/A
        Raises:
            N/A
        """
        # distribute C cash and A assets uniformly
        if dtype == DType.UNIFORM:
            # compute the number of cash and assets per agent
            cash_per = C / self.N
            asset_per = floor(A / self.N)
            for A in self.agents:
                A.C += cash_per
                A.A += asset_per

        # distribute C cash and A assets mostly to a few people and then
        # the reminder is divvy up among everyone else
        elif dtype == DType.ROYALTY:
            # separate the amount of cash and assets that belong to the royal group
            royal_c, royal_a = 0.85 * C, floor(0.85 * A)
            # get the amount of cash and assets to give to the rest
            remainder_c, remainder_a = C - royal_c, int(A - royal_a)
            # randomly select a small amount of agents to give most of the assets to
            royal_amount = int(0.10 * self.N)
            royalty_ind = np.random.choice(range(self.N), royal_amount, replace=False)
            # calculate how many cash and assets per royal
            cash_per_royal, asset_per_royal = royal_c / royal_amount, int(
                royal_a / royal_amount
            )
            # calculate how many cash and assets per non royal
            cash_per, asset_per = remainder_c / (self.N - royal_amount), int(
                remainder_a / (self.N - royal_amount)
            )
            # loop over agents
            for i in range(self.N):
                if i in royalty_ind:
                    self.agents[i].C += cash_per_royal
                    self.agents[i].A += asset_per_royal
                else:
                    self.agents[i].C += cash_per
                    self.agents[i].A += asset_per

    def cluster(self):
        """
        Computes the social network cluster and which (if any) are activated.

        Args:
            N/A
        Returns:
            N/A
        Raises:
            N/A
        """
        # make the social network
        self.make_network()
        # get connected components (this includes singletons)
        S = [
            self.network.subgraph(c).copy()
            for c in nx.connected_components(self.network)
            if len(c) > 1
        ]
        # see if we activate a cluster
        if (np.random.uniform() < self.pc) and (len(S) != 0):
            # now that a cluster is going to be activated we choose which cluster gets activated
            cluster = S[np.random.choice(range(len(S)))]
            # choose to activate this cluster as a sell or buy cluster via a coin flip
            new_p = float(np.random.uniform() <= 0.5)
            for n_id in cluster.nodes:
                # make the agent corresponding to the n_id node to buy or sell
                self.agents[n_id].prob = new_p

    def un_cluster(self):
        """
        Resets the social network by deactivating any active clusters and erases the edges.

        Args:
            N/A
        Returns:
            N/A
        Raises:
            N/A
        """
        # reset all the agents prob attribute to the original (this is 0.5 by default)
        for agent in self.agents:
            agent.prob = 0.5
        # reset the network edges
        self.reset_network()

    def make_orders(self):
        """
        This method uses the information of the market to make the agents order
        (either buy order or sell order).

        Args:
            N/A
        Returns:
            (buy_orders,sell_orders) :(numpy.array,numpy.array) - Each array in this tuple
            is a N-hx3 and hx3 array, each representing a
            [amount ordered,limit price, agent index] pair
        Raises:
            N/A
        """
        # draw a random value [0,1] per agent
        R = np.random.uniform(size=self.N)
        # draw a random value [0,1] to decide if agents will buy or sell
        buy_or_sell = np.random.uniform(size=self.N)

        # calculate the log price returns array
        # agents have built in methodology if its to short or long
        prices = np.array(self.prices)
        log_price_returns = np.log(prices[1:] / prices[:-1])

        # instantiate two list, each keeping track of buy and sell orders
        buy_orders = []
        sell_orders = []

        # loop over agents
        for i, agent in enumerate(self.agents):

            # decide if the agent will make a buy or sell order
            if buy_or_sell[i] < agent.prob:
                # make a buy order
                a_b, bi = agent.make_buy_order(
                    R[i], self.prices[-1], log_price_returns[-agent.Ti :]
                )
                buy_orders.append([a_b, bi, i])
            else:
                # make a sell order
                a_s, si = agent.make_sell_order(
                    R[i], self.prices[-1], log_price_returns[-agent.Ti :]
                )
                sell_orders.append([a_s, si, i])

        # return these two as Buy and Sell order arrays sorted by limit prices
        buy_orders = np.array(buy_orders)
        sell_orders = np.array(sell_orders)

        sorted_buy_orders = buy_orders[buy_orders[:, 1].argsort()]
        sorted_sell_orders = sell_orders[sell_orders[:, 1].argsort()]

        return sorted_buy_orders, sorted_sell_orders

    def market_price(self, BO, SO):
        """
        This method determines the Market price of the asset based on buy and sell orders.
        Appends the new market price calculation to the self.prices attribute.

        Args:
            BO :numpy.array - an (N-h)x2 array representing [buy amount,limit price,agent index]
                pairs
            SO :numpy.array - an (h)x2 array representing [sell amount,limit price,agent index]
                pairs
        Returns:
            N/A
        Raises:
            N/A
        """
        # calculate the demand and supply curve from buy and sell orders
        demand, supply = make_demand_supply(BO, SO)

        # get the combined domains of the supply and demand curves
        X = np.unique(np.hstack([demand.x, supply.x]))
        X = X[X.argsort()]

        # make sure to only keep the domain points within the common demand and supply domain
        xmin = max(demand.x[0], supply.x[0])
        xmax = min(demand.x[-1], supply.x[-1])
        X = X[(xmin <= X) & (X <= xmax)]

        # calculate the differences between demand sand supply in the common domain
        diffs = demand(X) - supply(X)
        # find the index where the sign changes
        idx = X[(np.argwhere(np.diff(np.sign(diffs))).flatten()) + 1]

        # case where a single price is identified
        if idx.shape == (1,):
            self.prices.append(idx[0])
        # case when theres a interval that the prices meet at
        elif idx.shape == (2,):
            self.prices.append(np.sum(idx) / 2)
        elif idx.shape == (0,):
            raise ValueError(f"idx shape is none at Age={self.age}")
        else:
            raise ValueError(f"i dont know shape at Age={self.age}")

    def execute_orders(self, BO, SO):
        """
        Executes the orders based on the latest market price (self.prices[-1]).

        Args:
            BO :numpy.array - an (N-h)x2 array representing [buy amount,limit price,agent index] pairs
            SO :numpy.array - an (h)x2 array representing [sell amount,limit price,agent index] pairs
        Returns:
            None
        Raises:
            ValueError("No Trades Possible")
        """
        # get the market price this time step
        pstar = self.prices[-1]
        # get Buy and Sell orders that fall within market price
        valid_buys = BO[BO[:, 1] >= pstar]
        valid_sells = SO[SO[:, 1] <= pstar]

        # we may need to get rid of some orders
        # if the sum of the stock in valid buys dont align with the sum of stocks
        # in valid sells
        delta = int(np.sum(valid_buys[:, 0]) - np.sum(valid_sells[:, 0]))
        if delta < 0:
            delta *= -1
            # in this case i need to get rid of delta sell orders
            select_rows = valid_sells
        elif delta > 0:
            # in this case  i need to get rid of delta buy orders
            select_rows = valid_buys

        # at this point we may not have any trades that we can make
        if delta != 0:
            if np.sum(select_rows[:, 0]) - delta == 0:
                raise ValueError(f"No Trades Possible at age of {self.age}")

        # loop over the number of orders to erase
        for _ in range(delta):
            rows, __ = select_rows.shape
            # choose a row weighted by the number of stocks ordered
            ps = select_rows[:, 0] / np.sum(select_rows[:, 0])
            row_index = np.random.choice(range(rows), p=ps)
            # now that this row is selected remove a stock from it
            select_rows[row_index, 0] -= 1

        # valid Buy and Sells have now been corrected so that there are k stocks to sell
        # and k stocks to buy
        for row in valid_buys:
            asset_amount, __, i = row
            i = int(i)
            price = asset_amount * pstar
            # the ith agent buys asset_amount for a totall of price
            self.agents[i].A += asset_amount
            self.agents[i].C -= price
        for row in valid_sells:
            asset_amount, __, i = row
            i = int(i)
            price = asset_amount * pstar
            # the ith agent sells asset_amount for a totall of price
            self.agents[i].A -= asset_amount
            self.agents[i].C += price

    def simulate(self, tau=10):
        """
        Simulating the market over tau steps. Simulation loop is:
            1) Make and activate social clusters
            2) computes buy and sell orders
            3) computes the new market price
            4) execute orders
            5) deactivates and removes clusters
            6) ages market

        Args:
            tau :int - the number of time steps to run
        Returns:
            N/A
        Raises:
            IndexError("No __ Orders Issued, Terminating Sim") when no orders are computed

        """
        # start a timer
        time_start = perf_counter()
        # run tau market steps
        for _ in range(tau):
            # make the clusters
            self.cluster()
            # calculate buy and sell orders
            buy_orders, sell_orders = self.make_orders()
            # check to see if one of the orders has weird shape
            if buy_orders.shape == (0,):
                raise IndexError("No Buy Orders Issued, Terminating Sim")
            elif sell_orders.shape == (0,):
                raise IndexError("No Sell Orders Issued, Terminating Sim")
            # calculate the market price of asset due to the orders
            self.market_price(buy_orders, sell_orders)
            # execute orders
            try:
                self.execute_orders(buy_orders, sell_orders)
            except ValueError as error:
                print(error)
            # reset clusters
            self.un_cluster()
            # age the market
            self.age += 1
        # end the timer
        time_end = perf_counter()
        sim_time = time_end - time_start
        # get time elapsed as string
        time = make_time_string(sim_time)
        print(
            f"Ran {tau} simulation steps for {self.N} agents over a period of " + time
        )
