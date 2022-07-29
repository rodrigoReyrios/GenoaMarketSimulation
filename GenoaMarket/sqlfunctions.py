"""This module hold various functions used to interact with an sql data base"""

import json
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Integer,
    Float,
    String,
    Column,
    ForeignKey,
)


def create_sqlite(data_base_name, **kwargs):
    """
    Makes an sql alchemy engine from an sqlite instance. This also runs a sql query that
    creates appropriate tables.

    Args:
        data_base_name :string - the name database to be used
        **kwargs - gets passed into the sqlalchemy.create_engine() function
    Returns:
        an sqlalchemy engine, a meta data object
    """
    # make the engine and pass the kwargs
    engine = create_engine("sqlite:///" + data_base_name, **kwargs)
    # since this will be a fresh sqlite data base Ill need to make the database
    # start by making a meta data
    meta_data = MetaData()

    # make an agents table
    agents = Table(
        "agents",
        meta_data,
        Column("agent_id", Integer, primary_key=True, autoincrement=False),
        Column("Ti", Integer),
        Column("k", Float(25)),
        Column("prob", Float(25)),
    )

    # make the market state table
    market_state = Table(
        "market_state",
        meta_data,
        Column("time_step", Integer, primary_key=True, autoincrement=False),
        Column("market_price", Float(25)),
        Column("buy_orders", Integer),
        Column("sell_orders", Integer),
    )

    # make an orders table
    orders = Table(
        "orders",
        meta_data,
        Column("order_type", String(4)),
        Column("order_amount", Integer),
        Column("limit_price", Float(25)),
        Column("order_id", Integer, primary_key=True, autoincrement=True),
        Column("agent_id", None, ForeignKey("agents.agent_id")),
        Column("time_step", None, ForeignKey("market_state.time_step")),
    )

    # make the portfolios table
    portfolios = Table(
        "portfolios",
        meta_data,
        Column("cash", Float(25)),
        Column("assets", Integer),
        Column(
            "agent_id",
            None,
            ForeignKey("agents.agent_id"),
            primary_key=True,
            autoincrement=False,
        ),
        Column(
            "time_step",
            None,
            ForeignKey("market_state.time_step"),
            primary_key=True,
            autoincrement=False,
        ),
    )

    # make all the tables
    meta_data.create_all(bind=engine)
    return engine, meta_data


def create_engine_from_file(json_path,data_base_name, **kwargs):
    """
    Makes an sql alchemy engine from a json file that stores all the data base link info.
    Note that the file is expected to be in form 'user name'.json.

    Args:
        json_path :string - the path of the json file
        data_base_name :string - the name of the database to acces
        **kwargs - gets passed into the sqlalchemy.create_engine() function
    Returns:
        an sqlalchemy engine, a brand new meta data
    """
    # get the users name
    user = json_path.split("/")[-1][:-5]

    with open(json_path, "rb") as read_file:
        sign_in = json.load(read_file)
    # create a db link based on the dicts parameters
    dialect, driver = sign_in["dialect"], sign_in["driver"]
    password = sign_in["pass"]
    host, port = sign_in["host"], sign_in["port"]
    # use the dictionary values to make the data base link
    link = f"{dialect}+{driver}://{user}:{password}@{host}:{port}/{data_base_name}"
    # use the link to make the engine
    new_engine = create_engine(link, **kwargs)
    return new_engine, MetaData()


def insert_agents(engine, agents_list, table_name="agents", meta_data=None):
    """
    Executes an insert query to store agent information. Expects the engine and tablename.
    Also note that the agents table (whatever its named) should have a topology like:

    CREATE TABLE agents(
        agent_id INT PRIMARY KEY,
        Ti INT,
        k FLOAT(25),
        prob FLOAT(25)
    )

    The column names are hard-coded in this function.

    Args:
        engine :sqlalchemy.engine - the engine connected to the sql database
        agent_list :list[Agent] - the list of agents to insert into the table
        table_name :str - the name of the table designated to hold the agents data
        meta_data :sqlalchemy.sql.schema.MetaData - a meta data object, if not provided
            will make a new one
    Returns:
        meta_data :sqlalchemy.sql.schema.MetaData - the meta data object that was either passed,
            or constructed within the function
    Raises:
        N/A

    """
    # construct the rows to insert
    rows = []
    for agent in agents_list:
        temp_dict = {
            "agent_id": agent.agent_id,
            "Ti": agent.Ti,
            "k": agent.k,
            "prob": agent.prob,
        }
        rows.append(temp_dict)

    # get the metadata, and table from the database and auto load with connection
    if meta_data is None:
        meta_data = MetaData()
    with engine.connect() as conn:
        # get the agents table
        agents_reflected = Table(table_name, meta_data, autoload_with=conn)
        # execute the insert
        conn.execute(agents_reflected.insert(), rows)
    return meta_data


def insert_market_state(
    engine,
    time_step,
    market_price=None,
    buy_orders=None,
    sell_orders=None,
    table_name="market_state",
    meta_data=None,
):
    """
    Executes an insert query to store a row for the market state. Expects an engine, the name of the
    table, and the row data. Also note that the market state table (whatever its named) should have
    a topology like:

    CREATE TABLE market_state(
        time_step INT PRIMARY KEY,
        market_price FLOAT(25),
        buy_orders INT,
        sell_orders INT
    )

    The column names are hard coded in this function.

    Args:
        engine :sqlalchemy.engine - the engine connected to the sql database
        time_step :int - the time step that this row is associated with.
        market_price :float - the price of the market at this time step
        buy_orders :int - the amount of buy orders that take place this time step
        sell_orders :int - the amount of sell orders that take place this time step
        table_name :str - the name of the market state table
        meta_data :sqlalchemy.sql.schema.MetaData - meta data object, if not provided will make new
    Returns:
        meta_data :sqlalchemy.sql.schema.MetaData - the meta data object that was either passed,
        or constructed within the function
    Raises:
        N/A
    """
    # make a meta data object if not given
    if meta_data is None:
        meta_data = MetaData()
    # make a connection on the engine and execute the row explicitly
    with engine.connect() as conn:
        # get the market_state table
        market_state_reflected = Table(table_name, meta_data, autoload_with=conn)
        # execute inserting the row
        conn.execute(
            market_state_reflected.insert().values(
                time_step=time_step,
                market_price=market_price,
                buy_orders=buy_orders,
                sell_orders=sell_orders,
            )
        )

    # return the meta data
    return meta_data


def insert_orders(
    engine, buy_orders, sell_orders, time_step, table_name="orders", meta_data=None
):
    """
    Executes an insert query to store orders that are made at a particular time step. Expects an
    engine, the table name, and the buy and sell orders arrays. Also note that the orders table
    (whatever its named) should have a topology like:

    CREATE TABLE orders(
        order_type VARCHAR(4),
        order_amount INT,
        limit_price FLOAT(25),
        order_id INT PRIMARY KEY AUTO_INCREMENT,
        agent_id INT,
        FOREIGN KEY(agent_id) REFERENCES agents(agent_id),
        time_step INT,
        FOREIGN KEY(time_step) REFERENCES market_state(time_step)
    )

    The column names are hard coded in this function.

    Args:
        engine :sqlalchemy.engine - the engine connected to the sql database
        buy_orders :numpy.array - an (N-h)x3 array representing [buy amount,limit price,agent index]
        sell_orders :numpy.array - an (h)x3 array representing [sell amount,limit price,agent index]
        time_step :int - the time step that this row is associated with.
        table_name :str - the name of the orders table
        meta_data :sqlalchemy.sql.schema.MetaData - meta data object, if not provided will make new
    Returns:
        meta_data :sqlalchemy.sql.schema.MetaData - the meta data object that was either passed,
            or constructed within the function
    Raises:
        N/A
    """
    # make a meta data object if one was not passed
    if meta_data is None:
        meta_data = MetaData()
    # i need to make a bunch of dictionaries to pass into the values of the insert query
    buy_rows = []
    for buy_amount, limit_price, agent_id in buy_orders:
        temp_dict = {
            "order_type": "buy",
            "order_amount": buy_amount,
            "limit_price": limit_price,
            "agent_id": agent_id,
            "time_step": time_step,
        }
        buy_rows.append(temp_dict)

    # do the same thing for sell orders
    sell_rows = []
    for sell_amount, limit_price, agent_id in sell_orders:
        temp_dict = {
            "order_type": "sell",
            "order_amount": sell_amount,
            "limit_price": limit_price,
            "agent_id": agent_id,
            "time_step": time_step,
        }
        sell_rows.append(temp_dict)

    # make a connection to the engine and insert the rows
    with engine.connect() as conn:
        # get the orders table
        orders_reflected = Table(table_name, meta_data, autoload_with=conn)
        # execute the insert
        conn.execute(orders_reflected.insert(), buy_rows)
        conn.execute(orders_reflected.insert(), sell_rows)
    # return the metadata
    return meta_data


def insert_portfolios(
    engine, agents_list, time_step, table_name="portfolios", meta_data=None
):
    """
    Executes an insert query to store portfolio data for each agent at a particular time step.
    Expects an engine, the table name, and the agents list. Also note that the portfolios
    table (whatever its named) should have a topology like:

    CREATE TABLE portfolios(
        agent_id INT,
        time_step INT,
        cash FLOAT,
        assets INT,
        FOREIGN KEY(agent_id) REFERENCES agents(agent_id),
        FOREIGN KEY(time_step) REFERENCES market_state(time_step),
        PRIMARY KEY(agent_id, time_step)
    )

    where agents and market_state are other tables. Note that it's required that agents and orders
    tables have the expected row entries and the column names are hard coded.

    Args:
        engine :sqlalchemy.engine - the engine connected to the sql database
        agent_list :list[Agent] - the list of agents who's portfolios to enter
        time_step :int - integer about what simulation step each portfolio entry is associated with
        table_name :str - the name of the table designated to hold the agents data
        meta_data :sqlalchemy.sql.schema.MetaData - meta data object, if not provided will make new
    Returns:
        meta_data :sqlalchemy.sql.schema.MetaData - the meta data object that was either passed,
            or constructed within the function
    Raises:
        N/A
    """
    # make a metadata if one is not passed
    if meta_data is None:
        meta_data = MetaData()
    # start making rows over
    portfolio_rows = []
    for agent in agents_list:
        temp_dict = {
            "agent_id": agent.agent_id,
            "time_step": time_step,
            "cash": agent.C,
            "assets": agent.A,
        }
        portfolio_rows.append(temp_dict)
    # open a connection and execute the insert statement
    with engine.connect() as conn:
        # get the orders table
        portfolios_reflected = Table(table_name, meta_data, autoload_with=conn)
        # execute the insert
        conn.execute(portfolios_reflected.insert(), portfolio_rows)
    # return the meta data object
    return meta_data
