#pragma once

#include <map>
#include <unordered_map>
#include <string>
#include <memory>
#include <boost/asio.hpp>

namespace arbitragex {

struct Order {
    std::string order_id;
    double price;
    int quantity;
    bool is_buy;
    std::string symbol;
};

class OrderBook {
public:
    OrderBook();
    ~OrderBook() = default;

    // Order management
    void add_order(const Order& order);
    void modify_order(const std::string& order_id, int new_quantity);
    void cancel_order(const std::string& order_id);

    // Market data
    std::map<double, int> get_bids() const;
    std::map<double, int> get_asks() const;
    double get_best_bid() const;
    double get_best_ask() const;
    double get_spread() const;

private:
    // Price level maps (price -> total quantity)
    std::map<double, int, std::greater<double>> bids_;
    std::map<double, int> asks_;
    
    // Order tracking
    std::unordered_map<std::string, Order> orders_;
    

    void update_price_levels(const Order& order, bool is_add);
    void remove_price_levels(const Order& order);
};

} 