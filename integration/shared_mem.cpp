#include <iostream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

class SharedMemory {
public:
    SharedMemory(const std::string& name, size_t size) 
        : shm_name_(name), size_(size), fd_(-1), data_(nullptr) {}
    
    ~SharedMemory() {
        disconnect();
    }
    
    bool connect() {
        // Create shared memory object
        fd_ = shm_open(shm_name_.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd_ == -1) {
            std::cerr << "Error creating shared memory: " << strerror(errno) << std::endl;
            return false;
        }
        
        // Set size
        if (ftruncate(fd_, size_) == -1) {
            std::cerr << "Error setting shared memory size: " << strerror(errno) << std::endl;
            close(fd_);
            return false;
        }
        
        // Map shared memory
        data_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (data_ == MAP_FAILED) {
            std::cerr << "Error mapping shared memory: " << strerror(errno) << std::endl;
            close(fd_);
            return false;
        }
        
        std::cout << "Connected to shared memory: " << shm_name_ << std::endl;
        return true;
    }
    
    void disconnect() {
        if (data_) {
            munmap(data_, size_);
            data_ = nullptr;
        }
        if (fd_ != -1) {
            close(fd_);
            fd_ = -1;
        }
    }
    
    bool write_order(double price, int quantity, bool is_buy) {
        if (!data_) {
            std::cerr << "Not connected to shared memory" << std::endl;
            return false;
        }
        
        // Pack order data
        struct OrderData {
            double price;
            int quantity;
            int is_buy;
        } order = {price, quantity, is_buy ? 1 : 0};
        
        // Write to shared memory
        memcpy(data_, &order, sizeof(order));
        return true;
    }
    
    bool read_market_data(double& best_bid, double& best_ask, double& last_price) {
        if (!data_) {
            std::cerr << "Not connected to shared memory" << std::endl;
            return false;
        }
        
        // Read from shared memory
        struct MarketData {
            double best_bid;
            double best_ask;
            double last_price;
        } market_data;
        
        memcpy(&market_data, data_, sizeof(market_data));
        
        best_bid = market_data.best_bid;
        best_ask = market_data.best_ask;
        last_price = market_data.last_price;
        
        return true;
    }
    
private:
    std::string shm_name_;
    size_t size_;
    int fd_;
    void* data_;
};

int main() {
    // Example usage
    SharedMemory shm("arbitragex_shm", 1024 * 1024);
    
    if (!shm.connect()) {
        return 1;
    }
    
    try {
        // Write some orders
        shm.write_order(100.0, 100, true);  // Buy 100 at 100.0
        usleep(100000);  // Sleep for 100ms
        shm.write_order(101.0, 50, false);  // Sell 50 at 101.0
        
        // Read market data
        double best_bid, best_ask, last_price;
        if (shm.read_market_data(best_bid, best_ask, last_price)) {
            std::cout << "Market Data:" << std::endl;
            std::cout << "Best Bid: " << best_bid << std::endl;
            std::cout << "Best Ask: " << best_ask << std::endl;
            std::cout << "Last Price: " << last_price << std::endl;
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 