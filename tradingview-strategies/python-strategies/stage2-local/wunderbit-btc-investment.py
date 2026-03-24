#!/usr/bin/env python3
import numpy as np
import pandas as pd

name = "Automated Bitcoin (BTC) Investment Strategy from Wunderbit"
timeframe = "4h"
leverage = 1

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing method."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr

def calculate_ema(series, length):
    """Calculate Exponential Moving Average."""
    if length <= 0:
        return np.full(len(series), np.nan)
    ema = np.zeros(len(series))
    ema[0] = series[0]
    multiplier = 2 / (length + 1)
    for i in range(1, len(series)):
        ema[i] = (series[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema

def calculate_tema(series, length):
    """Calculate Triple Exponential Moving Average."""
    if length <= 0:
        return np.full(len(series), np.nan)
    ema1 = calculate_ema(series, length)
    ema2 = calculate_ema(ema1, length)
    ema3 = calculate_ema(ema2, length)
    return 3 * ema1 - 3 * ema2 + ema3

def calculate_lsma(series, length):
    """Calculate Least Squares Moving Average (linear regression)."""
    n = len(series)
    lsma = np.full(n, np.nan)
    for i in range(length - 1, n):
        x = np.arange(length)
        y = series[i - length + 1:i + 1]
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        if denominator != 0:
            slope = numerator / denominator
            intercept = y_mean - slope * x_mean
            lsma[i] = slope * (length - 1) + intercept
    return lsma

def calculate_trend_line(close, trend_type, length):
    """Calculate trend line based on type."""
    if trend_type == 'TEMA':
        return calculate_tema(close, length)
    elif trend_type == 'LSMA':
        return calculate_lsma(close, length)
    elif trend_type == 'EMA':
        return calculate_ema(close, length)
    else:  # SMA
        return pd.Series(close).rolling(window=length, min_periods=1).mean().values

def generate_signals(prices):
    """
    Generate target position signals based on Pine Script strategy logic.
    Returns numpy array with target position fractions (0 or 1 for long-only).
    """
    n = len(prices)
    if n == 0:
        return np.zeros(n, dtype=np.float64)
    
    close = prices['close'].values.astype(np.float64)
    high = prices['high'].values.astype(np.float64)
    low = prices['low'].values.astype(np.float64)
    
    # Trend lines (default: TEMA 25, LSMA 100)
    leadLine1 = calculate_trend_line(close, 'TEMA', 25)
    leadLine2 = calculate_trend_line(close, 'LSMA', 100)
    
    # ATR for trailing stop
    atr = calculate_atr(high, low, close, 8)
    multiplier = 3.5
    
    # Calculate trailing stop (stateful logic requires loop)
    trail1 = np.zeros(n)
    sl1 = multiplier * atr
    
    for i in range(n):
        if i == 0:
            trail1[i] = close[i]
        else:
            prev_trail = trail1[i-1] if not np.isnan(trail1[i-1]) else close[i]
            if close[i] > prev_trail:
                trail1[i] = close[i] - sl1[i]
            elif close[i] < prev_trail and close[i-1] < prev_trail:
                trail1[i] = min(prev_trail, close[i] + sl1[i])
            else:
                trail1[i] = close[i] + sl1[i]
    
    # Trail1 high (50-period highest)
    trail1_high = pd.Series(trail1).rolling(window=50, min_periods=1).max().values
    
    # Entry/Exit conditions
    # Crossover: leadLine1 crosses above leadLine2
    crossover = (leadLine1 > leadLine2) & (np.roll(leadLine1, 1) <= np.roll(leadLine2, 1))
    crossover[0] = False  # No crossover on first bar
    
    # Crossunder: leadLine1 crosses below leadLine2
    crossunder = (leadLine1 < leadLine2) & (np.roll(leadLine1, 1) >= np.roll(leadLine2, 1))
    crossunder[0] = False
    
    # TP/SL levels (based on entry price, tracked in loop)
    long_tp1_pct = 0.15
    long_tp2_pct = 0.30
    long_sl_pct = 0.05
    
    # Generate signals with stateful position tracking
    signals = np.zeros(n, dtype=np.float64)
    in_position = False
    entry_price = 0.0
    
    for i in range(n):
        # Calculate dynamic TP/SL levels based on entry price
        if in_position and entry_price > 0:
            long_take_level_1 = entry_price * (1 + long_tp1_pct)
            long_take_level_2 = entry_price * (1 + long_tp2_pct)
            long_sl_level = entry_price * (1 - long_sl_pct)
        else:
            long_take_level_1 = 0
            long_take_level_2 = 0
            long_sl_level = 0
        
        # Entry condition: crossover and price above trailing stop high
        entry_long = crossover[i] and (close[i] > trail1_high[i])
        
        # Exit conditions
        exit_long = (
            close[i] < trail1_high[i] or
            crossunder[i] or
            (in_position and close[i] < long_sl_level)
        )
        
        # Take profit conditions (partial exits simulated as full exit for simplicity)
        tp_exit = False
        if in_position and entry_price > 0:
            if close[i] >= long_take_level_2:
                tp_exit = True
            elif close[i] >= long_take_level_1:
                tp_exit = True
        
        if not in_position and entry_long:
            signals[i] = 1.0
            in_position = True
            entry_price = close[i]
        elif in_position and (exit_long or tp_exit):
            signals[i] = 0.0
            in_position = False
            entry_price = 0.0
        else:
            signals[i] = 1.0 if in_position else 0.0
    
    return signals
