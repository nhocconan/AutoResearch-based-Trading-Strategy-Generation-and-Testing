#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA trend following with 1d RSI filter and volume confirmation.
# Long when KAMA indicates uptrend AND 1d RSI > 50 AND volume > 1.3x 20-period average.
# Short when KAMA indicates downtrend AND 1d RSI < 50 AND volume > 1.3x 20-period average.
# Exit when KAMA reverses direction.
# KAMA adapts to market noise, reducing whipsaw in sideways markets.
# 1d RSI ensures alignment with higher timeframe momentum.
# Volume confirmation filters low-conviction moves.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_KAMA_1dRSI_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    fast_ema = 2
    slow_ema = 30
    lookback = 10
    
    # Calculate ER (Efficiency Ratio)
    change = np.abs(np.diff(close, n=lookback))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Calculate SSC (Smoothing Constant)
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA direction: 1 for up, -1 for down
    kama_dir = np.zeros(n, dtype=int)
    kama_dir[1:] = np.where(kama[1:] > kama[:-1], 1, -1)
    
    # 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d RSI(14)
    rsi_period = 14
    delta = np.diff(df_1d['close'].values)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(delta)
    avg_loss = np.zeros_like(delta)
    avg_gain[rsi_period-1] = np.mean(gain[:rsi_period])
    avg_loss[rsi_period-1] = np.mean(loss[:rsi_period])
    
    for i in range(rsi_period, len(delta)):
        avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
        avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Pad RSI to match close array length
    rsi_full = np.full(n, np.nan)
    rsi_full[lookback:] = rsi[rsi_period-1:]
    
    # Align RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_full)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, rsi_period, 20)  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_dir[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: KAMA up, 1d RSI > 50, volume filter
            long_cond = (kama_dir[i] == 1) and (rsi_aligned[i] > 50) and volume_filter[i]
            # Short conditions: KAMA down, 1d RSI < 50, volume filter
            short_cond = (kama_dir[i] == -1) and (rsi_aligned[i] < 50) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: KAMA turns down
            if kama_dir[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: KAMA turns up
            if kama_dir[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals