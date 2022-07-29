"""Module containing basic Agent class"""

from dataclasses import dataclass, field
from math import floor
import warnings
import numpy as np
from .constant import SMALL_HISTORY_WARNING
from .constant import NEGATIVE_PRICE_LIMIT_SCALE_WARNING


@dataclass
class Agent:
    """
    Class Representation of a trader Agent.

    Attributes
    ----------
    C :float = 50.0
        The number of cash this agent has
    A :float = 10.0
        The number of assets this agent holds
    Ti :int = 20
        The working 'memory' the agent has, used in calculating log price returns
    prob :float = 0.5
        The probability this agent puts a buy order (1-prob is the sell order probability)
    k :float = 3.5
        The constant that determines the spread of the agent chooses limit prices
    agent_id :int = -1
        The id of the agent, really just for print outs

    """

    C: float = field(default=50.0)
    A: float = field(default=10.0)
    Ti: int = field(default=20)
    prob: float = field(default=0.5)
    k: float = field(default=3.5)
    agent_id: int = field(default=-1)

    def make_buy_order(self, ri, pi, log_prices):
        """
        Instructs agent to make a buy order.

        Args:
            ri :float - the proportion of cash the agent wants to spend.
            pi :float - the current price of the asset.
            log_prices :list[float] - the list of prices the agent will consider when making order.
        Returns:
            [a_b, bi] :list - array symbolizing [quantity ordered, limit price]
        Raises:
        """
        # decide on the quantity of cash agent wants to buy stock with
        ci = ri * self.C
        # calculate the price limit scaling value
        price_scale = self.price_limit_scale(log_prices)
        # now that the price scale has been selected the buy limit price can be calculated
        bi = pi * price_scale

        a_b = floor(ci / bi)

        return [a_b, bi]

    def make_sell_order(self, ri, pi, log_prices):
        """
        Instructs agent to make a sell order
        Args:
            ri :float - the proportion of asset that the agent wants to put up for sale.
            pi :float - the current price of the asset.
            log_prices :list[float] - the list of log price returns the agent will consider when
                making order.
        Returns:
            [a_s, si] :list - array symbolizing [quantity for sale, limit price]
        Raises:
        """
        # decide on the quantity of stock the agent wants to sell
        a_s = floor(ri * self.A)
        # calculate the price limit scaling value
        price_scale = self.price_limit_scale(log_prices)
        # now that price scale has been chosen calculate sell limit
        si = pi / (price_scale)
        return [a_s, si]

    def make_order(self, ri, pi, log_prices, buy_or_sell):
        """
        Wrapper for making a buy or sell order by using the same parameters and a boolean.

        Args:
            ri :float - the proportion of asset/cash that the agent wants to put up for sale/buy.
            pi :float - the current price of the asset.
            log_prices :list[float] - the list of log price returns the agent will consider when
                making order.
            buy_or_sell :boolean - if true executes a buy order if false executes a sell order
        Returns:
            [a, i] :list - array symbolizing [quantity, limit price]
        Raises:
            N/A
        """
        order_list = None
        if buy_or_sell:
            order_list = self.make_buy_order(ri, pi, log_prices)
        else:
            order_list = self.make_sell_order(ri, pi, log_prices)

        return order_list

    def price_limit_scale(self, log_prices):
        """
        Draws a scalar value from a normal distribution defined by mu=1.01 and
        sigma = k*std(log_prices), where log_prices is meant to be an array of log price returns.
        Since there are various possible edge cases for when len(log_prices)<3, or maybe if
        an invalid scalar is drawn, this function is meant to control edge and
        special cases.

        Args:
            log_prices :list - the array of log price returns this agent remembers.
        Returns:
            price_limit_scale :float
        Raises:
            N/A
        """
        # initialize a float
        pre_sigma = 0.0
        # check if the agent is remembering less than 3 log price returns, this is considered
        # to small a history
        if len(log_prices) < 3:
            if SMALL_HISTORY_WARNING:
                warnings.warn(
                    f"Agent {self.agent_id} is given to small of a history of log price returns (<3), will set sigma (pre scaling by k) to a default value"
                )
            # when history is to small set sigma to some default value
            pre_sigma = 1 / self.k
        else:
            # otherwise we just get the std of the log price returns in memory
            pre_sigma = np.std(log_prices)
        # scale pre_sigma by the k constant, this is the actual sigma of the normal dist
        sigma = self.k * pre_sigma
        # draw a value from a normal distribution defined by mu=1.01, sigma=sigma
        price_limit_scale = np.random.normal(loc=1.01, scale=sigma)

        # it could be the case that the drawn price_limit_scale is negative
        if price_limit_scale < 0:
            # if we draw a negative value reverse it
            price_limit_scale *= -1
            if NEGATIVE_PRICE_LIMIT_SCALE_WARNING:
                warnings.warn(
                    f"Agent {self.agent_id} has drawn a negative value to determine price_limit_scale. Has been corrected to {price_limit_scale}"
                )

        return price_limit_scale
