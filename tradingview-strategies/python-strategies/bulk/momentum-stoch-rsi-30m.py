#!/usr/bin/env python3
"""
Momentum Strategy (BTC/USDT; 30m) - STOCH RSI
Converted from TradingView Pine Script
"""

import numpy as np
import pandas as pd

name = "Momentum Strategy (BTC/USDT; 30m) - STOCH RSI"
timeframe = "30m"
leverage = 1

# Strategy parameters
RSI_LENGTH = 12
STOCH_LENGTH = 12
SMOOTH_K = 3
SMOOTH_D = 6
FAST_EMA = 1
SLOW_EMA = 60
BARS_DELAY = 6
OVERBOUGHT = 85.29
OVERSOLD = 30.6
SL_LONG_PCT = 10.0
TP_LONG_PCT = 8.0
SL_SHORT_PCT = 20.0
TP_SHORT_PCT = 35.0


def calculate_rsi(close, length):
    """Calculate RSI using pandas/numpy only"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # Initial SMA
    avg_gain[:length] = np.mean(gain[:length]) if length <= len(gain) else np.mean(gain)
    avg_loss[:length] = np.mean(loss[:length]) if length <= len(loss) else np.mean(loss)
    
    # EMA for rest
    for i in range(length, len(close)):
        avg_gain[i] = (avg_gain[i-1] * (length - 1) + gain[i]) / length
        avg_loss[i] = (avg_loss[i-1] * (length - 1) + loss[i]) / length
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def calculate_stoch_rsi(rsi, length):
    """Calculate Stochastic of RSI"""
    lowest_rsi = np.zeros_like(rsi)
    highest_rsi = np.zeros_like(rsi)
    
    for i in range(len(rsi)):
        start_idx = max(0, i - length + 1)
        lowest_rsi[i] = np.min(rsi[start_idx:i+1])
        highest_rsi[i] = np.max(rsi[start_idx:i+1])
    
    stoch_rsi = np.zeros_like(rsi)
    range_rsi = highest_rsi - lowest_rsi
    mask = range_rsi != 0
    stoch_rsi[mask] = 100.0 * (rsi[mask] - lowest_rsi[mask]) / range_rsi[mask]
    
    return stoch_rsi


def calculate_ema(data, length):
    """Calculate EMA"""
    ema = np.zeros_like(data)
    ema[0] = data[0]
    multiplier = 2.0 / (length + 1.0)
    
    for i in range(1, len(data)):
        ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
    
    return ema


def generate_signals(prices):
    """
    Generate trading signals based on Stoch RSI momentum strategy.
    
    Args:
        prices: pandas DataFrame with columns: open_time, open, high, low, close, volume
    
    Returns:
        numpy array with signals: 0 (hold), 1 (long), -1 (short)
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.int8)
    
    close = prices['close'].values.astype(np.float64)
    high = prices['high'].values.astype(np.float64)
    low = prices['low'].values.astype(np.float64)
    
    # Calculate indicators
    rsi = calculate_rsi(close, RSI_LENGTH)
    stoch_rsi = calculate_stoch_rsi(rsi, STOCH_LENGTH)
    k = calculate_ema(stoch_rsi, SMOOTH_K)
    d = calculate_ema(k, SMOOTH_D)
    
    ema_fast = calculate_ema(close, FAST_EMA)
    ema_slow = calculate_ema(close, SLOW_EMA)
    
    # Track position state
    position = 0  # 0: none, 1: long, -1: short
    entry_price = 0.0
    bars_since_close_signal = 0
    last_trade_was_loss = False
    current_size_pct = 70.0  # Base order size percent
    
    # Detect crosses
    for i in range(1, n):
        golden_cross = (k[i-1] <= d[i-1]) and (k[i] > d[i])
        death_cross = (k[i-1] >= d[i-1]) and (k[i] < d[i])
        
        in_uptrend = ema_fast[i] > ema_slow[i]
        in_downtrend = ema_fast[i] < ema_slow[i]
        
        # Long entry conditions
        long_condition = golden_cross and (d[i] <= OVERSOLD)
        if long_condition and in_uptrend and position == 0:
            position = 1
            entry_price = close[i]
            signals[i] = 1
            last_trade_was_loss = False
            current_size_pct = 70.0
        
        # Short entry conditions
        short_condition = death_cross and (d[i] >= OVERBOUGHT)
        if short_condition and in_downtrend and position == 0:
            position = -1
            entry_price = close[i]
            signals[i] = -1
            last_trade_was_loss = False
            current_size_pct = 70.0
        
        # Close conditions with bars delay
        close_long_signal = death_cross and (d[i] >= OVERBOUGHT)
        close_short_signal = golden_cross and (d[i] <= OVERSOLD)
        
        if position == 1:
            if close_long_signal:
                bars_since_close_signal += 1
            else:
                bars_since_close_signal = 0
            
            # Check SL/TP levels (next-bar approximation)
            sl_level = entry_price * (1.0 - SL_LONG_PCT / 100.0)
            tp_level = entry_price * (1.0 + TP_LONG_PCT / 100.0)
            
            # Check if SL/TP hit using high/low of current bar
            sl_hit = low[i] <= sl_level
            tp_hit = high[i] >= tp_level
            
            if sl_hit or tp_hit or (bars_since_close_signal >= BARS_DELAY and close_long_signal):
                signals[i] = 0
                last_trade_was_loss = sl_hit
                if last_trade_was_loss and current_size_pct < 100.0:
                    current_size_pct = min(current_size_pct + 25.0, 100.0)
                else:
                    current_size_pct = 70.0
                position = 0
                bars_since_close_signal = 0
        
        elif position == -1:
            if close_short_signal:
                bars_since_close_signal += 1
            else:
                bars_since_close_signal = 0
            
            # Check SL/TP levels (next-bar approximation)
            sl_level = entry_price * (1.0 + SL_SHORT_PCT / 100.0)
            tp_level = entry_price * (1.0 - TP_SHORT_PCT / 100.0)
            
            # Check if SL/TP hit using high/low of current bar
            sl_hit = high[i] >= sl_level
            tp_hit = low[i] <= tp_level
            
            if sl_hit or tp_hit or (bars_since_close_signal >= BARS_DELAY and close_short_signal):
                signals[i] = 0
                last_trade_was_loss = sl_hit
                if last_trade_was_loss and current_size_pct < 100.0:
                    current_size_pct = min(current_size_pct + 25.0, 100.0)
                else:
                    current_size_pct = 70.0
                position = 0
                bars_since_close_signal = 0
    
    return signals


if __name__ == "__main__":
    # Test with sample data
    import pandas as pd
    
    # Create sample OHLCV data
    np.random.seed(42)
    n_samples = 1000
    base_price = 50000
    
    returns = np.random.randn(n_samples) * 0.02
    close = base_price * np.cumprod(1 + returns)
    open_prices = np.roll(close, 1)
    open_prices[0] = base_price
    high = np.maximum(open_prices, close) * (1 + np.abs(np.random.randn(n_samples)) * 0.01)
    low = np.minimum(open_prices, close) * (1 - np.abs(np.random.randn(n_samples)) * 0.01)
    volume = np.random.randint(1000, 10000, n_samples)
    open_time = np.arange(n_samples) * 1800000  # 30m in ms
    
    df = pd.DataFrame({
        'open_time': open_time,
        'open': open_prices,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    signals = generate_signals(df)
    print(f"Generated {len(signals)} signals")
    print(f"Long signals: {np.sum(signals == 1)}")
    print(f"Short signals: {np.sum(signals == -1)}")
    print(f"Signal array length matches prices: {len(signals) == len(df)}")
