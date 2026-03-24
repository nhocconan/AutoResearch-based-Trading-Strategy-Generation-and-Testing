#!/usr/bin/env python3
"""
Automated Bitcoin (BTC) Investment Strategy from Wunderbit
Converted from TradingView Pine Script to Python

Timeframe: 4h
Strategy Type: Trend-following with trailing stop
"""

import numpy as np
import pandas as pd

name = "Automated Bitcoin (BTC) Investment Strategy from Wunderbit"
timeframe = "4h"
leverage = 1


def ema(series, length):
    """Calculate Exponential Moving Average"""
    if length <= 0 or len(series) == 0:
        return np.full_like(series, np.nan, dtype=float)
    
    result = np.zeros(len(series), dtype=float)
    result[:] = np.nan
    
    multiplier = 2.0 / (length + 1)
    
    # Initialize with SMA for first valid value
    if length <= len(series):
        result[length - 1] = np.mean(series[:length])
        
        # Calculate EMA for remaining values
        for i in range(length, len(series)):
            result[i] = (series[i] - result[i - 1]) * multiplier + result[i - 1]
    
    return result


def tema(series, length):
    """Calculate Triple Exponential Moving Average"""
    if length <= 0:
        return np.full_like(series, np.nan, dtype=float)
    
    ema1 = ema(series, length)
    ema2 = ema(ema1, length)
    ema3 = ema(ema2, length)
    
    return 3 * ema1 - 3 * ema2 + ema3


def sma(series, length):
    """Calculate Simple Moving Average"""
    if length <= 0:
        return np.full_like(series, np.nan, dtype=float)
    
    result = np.zeros(len(series), dtype=float)
    result[:] = np.nan
    
    for i in range(length - 1, len(series)):
        result[i] = np.mean(series[i - length + 1:i + 1])
    
    return result


def lsma(series, length):
    """Calculate Least Squares Moving Average (Linear Regression)"""
    if length <= 0:
        return np.full_like(series, np.nan, dtype=float)
    
    result = np.zeros(len(series), dtype=float)
    result[:] = np.nan
    
    for i in range(length - 1, len(series)):
        window = series[i - length + 1:i + 1]
        x = np.arange(length)
        x_mean = np.mean(x)
        y_mean = np.mean(window)
        
        numerator = np.sum((x - x_mean) * (window - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator != 0:
            slope = numerator / denominator
            intercept = y_mean - slope * x_mean
            result[i] = slope * (length - 1) + intercept
        else:
            result[i] = y_mean
    
    return result


def calculate_atr(high, low, close, period):
    """Calculate Average True Range using Wilder's smoothing"""
    n = len(close)
    if n == 0 or period <= 0:
        return np.zeros(n, dtype=float)
    
    tr = np.zeros(n, dtype=float)
    
    # First TR is just high - low
    tr[0] = high[0] - low[0]
    
    # Calculate True Range for remaining bars
    for i in range(1, n):
        high_low = high[i] - low[i]
        high_prev_close = abs(high[i] - close[i - 1])
        low_prev_close = abs(low[i] - close[i - 1])
        tr[i] = max(high_low, high_prev_close, low_prev_close)
    
    # Wilder's smoothing for ATR
    atr = np.zeros(n, dtype=float)
    
    if period <= n:
        # First ATR is simple average of first 'period' TR values
        atr[period - 1] = np.mean(tr[:period])
        
        # Apply Wilder's smoothing
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_trend_line(close, trend_type, length):
    """Calculate trend line based on specified type"""
    if trend_type == "TEMA":
        return tema(close, length)
    elif trend_type == "EMA":
        return ema(close, length)
    elif trend_type == "SMA":
        return sma(close, length)
    elif trend_type == "LSMA":
        return lsma(close, length)
    else:
        return ema(close, length)


def generate_signals(prices):
    """
    Generate trading signals based on price data.
    
    Args:
        prices: pandas DataFrame with columns:
                open_time, open, high, low, close, volume
    
    Returns:
        numpy array with exactly len(prices) elements:
        1 = Long position, 0 = Flat/No position
    """
    # Extract OHLCV data as numpy arrays
    open_price = prices["open"].values.astype(float)
    high = prices["high"].values.astype(float)
    low = prices["low"].values.astype(float)
    close = prices["close"].values.astype(float)
    
    n = len(close)
    if n == 0:
        return np.array([], dtype=int)
    
    # Strategy parameters (from Pine Script defaults)
    trend_type1 = "TEMA"
    trend_type2 = "LSMA"
    trend_type1_length = 25
    trend_type2_length = 100
    atr_period = 8
    multiplier = 3.5
    long_tp1_pct = 0.15
    long_tp2_pct = 0.30
    long_sl_pct = 0.05
    
    # Calculate trend lines
    lead1 = calculate_trend_line(close, trend_type1, trend_type1_length)
    lead2 = calculate_trend_line(close, trend_type2, trend_type2_length)
    
    # Calculate ATR
    atr = calculate_atr(high, low, close, atr_period)
    
    # Calculate trailing stop (stateful logic)
    trail1 = np.zeros(n, dtype=float)
    trail1_high = np.zeros(n, dtype=float)
    
    for i in range(n):
        sl1 = multiplier * atr[i] if i < n else multiplier * atr[-1]
        
        if i == 0:
            trail1[i] = close[i] + sl1
        else:
            prev_trail = trail1[i - 1]
            
            if close[i] > prev_trail and close[i - 1] > prev_trail:
                trail1[i] = close[i] - sl1
            elif close[i] < prev_trail and close[i - 1] < prev_trail:
                trail1[i] = min(prev_trail, close[i] + sl1)
            else:
                if close[i] > prev_trail:
                    trail1[i] = close[i] - sl1
                else:
                    trail1[i] = close[i] + sl1
        
        # Calculate 50-bar highest of Trail1
        start_idx = max(0, i - 49)
        trail1_high[i] = np.max(trail1[start_idx:i + 1])
    
    # Detect crossovers and crossunders
    crossover_1_2 = np.zeros(n, dtype=bool)
    crossunder_2_1 = np.zeros(n, dtype=bool)
    
    for i in range(1, n):
        if not np.isnan(lead1[i]) and not np.isnan(lead2[i]):
            if not np.isnan(lead1[i - 1]) and not np.isnan(lead2[i - 1]):
                if lead1[i - 1] <= lead2[i - 1] and lead1[i] > lead2[i]:
                    crossover_1_2[i] = True
                if lead1[i - 1] >= lead2[i - 1] and lead1[i] < lead2[i]:
                    crossunder_2_1[i] = True
    
    # Entry condition: crossover and close below Trail1_high
    entry_long = np.zeros(n, dtype=bool)
    for i in range(n):
        if crossover_1_2[i] and close[i] < trail1_high[i]:
            if not np.isnan(trail1_high[i]):
                entry_long[i] = True
    
    # Initialize position tracking
    signals = np.zeros(n, dtype=int)
    in_position = False
    entry_price = 0.0
    tp1_level = 0.0
    tp2_level = 0.0
    sl_level = 0.0
    
    # Warmup period - need enough data for indicators
    warmup = max(trend_type1_length, trend_type2_length, atr_period, 50)
    
    for i in range(n):
        # Skip warmup period
        if i < warmup:
            signals[i] = 0
            continue
        
        # Entry logic
        if entry_long[i] and not in_position:
            # Entry executes at next bar open (approximation)
            if i + 1 < n:
                entry_price = open_price[i + 1]
            else:
                entry_price = close[i]
            
            tp1_level = entry_price * (1 + long_tp1_pct)
            tp2_level = entry_price * (1 + long_tp2_pct)
            sl_level = entry_price * (1 - long_sl_pct)
            
            in_position = True
            signals[i] = 1
        
        # Exit logic when in position
        elif in_position:
            exit_condition = False
            
            # Trailing stop exit
            if close[i] < trail1_high[i]:
                exit_condition = True
            
            # Trend crossunder exit
            if crossunder_2_1[i]:
                exit_condition = True
            
            # Stop loss exit
            if close[i] < sl_level:
                exit_condition = True
            
            # Take profit exits (approximated with close price)
            if close[i] >= tp1_level or close[i] >= tp2_level:
                exit_condition = True
            
            if exit_condition:
                signals[i] = 0
                in_position = False
                entry_price = 0.0
    
    return signals


if __name__ == "__main__":
    # Example usage
    print(f"Strategy: {name}")
    print(f"Timeframe: {timeframe}")
    print(f"Leverage: {leverage}")
