"""Miscellaneous functions used throughout a couple modules"""
from enum import Enum
import datetime
import numpy as np
from scipy.interpolate import interp1d

make_time_string = lambda x: str(datetime.timedelta(seconds=x))


def make_demand_supply(buy_orders, sell_orders):
    """
    function responsible for constructing demand and supply curves based on step
    wise interpolation using scipy.interp1d.

    Args:
        buy_orders
        sell_orders
    Returns:
        demand,supply :(scipy.interp1d,scipy.interp1d)
    Raises:
        N/A
    """
    # interpolation arguments
    args = {"kind": "previous"}

    # calculate the demand and supply curves via cum sum and reverse cum sum on assets ordered
    sup_x, sup_y = sell_orders[:, 1], np.cumsum(sell_orders[:, 0])
    dem_x, dem_y = buy_orders[:, 1], np.cumsum(buy_orders[:, 0][::-1])[::-1]

    # extend the supply curve to the right on any x values covered in the demands x's
    sup_x_right = dem_x[sup_x[-1] < dem_x]
    sup_y_right = np.full(sup_x_right.shape, sup_y[-1])

    # extend the supply curve to the left on any x values covered in the demands x's
    sup_x_left = dem_x[dem_x < sup_x[0]]
    sup_y_left = np.full(sup_x_left.shape, 0)

    # we also prepend a zero to the supply that equals 0

    # make an interpolated step function
    supply = interp1d(
        np.concatenate([[0], sup_x_left, sup_x, sup_x_right]),
        np.concatenate([[0], sup_y_left, sup_y, sup_y_right]),
        **args,
    )

    # extend the demand curve to the left on any x values covered in the supply's x's
    dem_x_left = sup_x[sup_x < dem_x[0]]
    dem_y_left = np.full(dem_x_left.shape, dem_y[0])
    # extend the demand curve to the right on any x values covered in the supply's x's
    dem_x_right = sup_x[dem_x[-1] < sup_x]
    dem_y_right = np.full(dem_x_right.shape, 0.0)

    # we also prepend a zero to the demand that equals the first value in the demands y
    # this value may come from dem_y or dem_y_left
    dem_zero = []
    if dem_y_left.shape == (0,):
        dem_zero.append(dem_y[0])
    else:
        dem_zero.append(dem_y_left[0])

    # make an interpolated step function
    demand = interp1d(
        np.concatenate([[0], dem_x_left, dem_x, dem_x_right]),
        np.concatenate([dem_zero, dem_y_left, dem_y, dem_y_right]),
        **args,
    )

    return demand, supply


class DType(Enum):
    """
    Simple child of Enum, meant to strong type which distribution to use when divvying up
    Cash and Assets to the market.
    """

    UNIFORM = "Uniform"
    ROYALTY = "Royalty"
