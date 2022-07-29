-- DROP TABLE IF EXISTS portfolios;
-- DROP TABLE IF EXISTS orders;
-- DROP TABLE IF EXISTS market_state;
-- DROP TABLE IF EXISTS agents;
-- agents table
CREATE TABLE agents(
    agent_id INT PRIMARY KEY,
    Ti INT,
    k FLOAT(25),
    prob FLOAT(25)
);
-- market state table
CREATE TABLE market_state(
    time_step INT PRIMARY KEY,
    market_price FLOAT(25),
    buy_orders INT,
    sell_orders INT
);
-- orders table
CREATE TABLE orders(
    order_type VARCHAR(4),
    order_amount INT,
    limit_price FLOAT(25),
    order_id INT PRIMARY KEY AUTO_INCREMENT,
    agent_id INT,
    FOREIGN KEY(agent_id) REFERENCES agents(agent_id),
    time_step INT,
    FOREIGN KEY(time_step) REFERENCES market_state(time_step)
);
-- make protfolio table
CREATE TABLE portfolios(
    agent_id INT,
    time_step INT,
    cash FLOAT(25),
    assets INT,
    PRIMARY KEY (agent_id, time_step),
    FOREIGN KEY(agent_id) REFERENCES agents(agent_id),
    FOREIGN KEY(time_step) REFERENCES market_state(time_step)
);