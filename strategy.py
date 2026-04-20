#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h chart with 1w Fibonacci retracement levels and 1d trend filter.
# Long when price retraces to 61.8% Fib level in uptrend (price > 200 EMA) with volume confirmation.
# Short when price retraces to 38.2% Fib level in downtrend (price < 200 EMA) with volume confirmation.
# Uses weekly swing high/low for Fib levels and daily EMA200 for trend filter.
# Target: 15-30 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200)
    
    # Load 1w data for Fibonacci levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly swing high and low (50-period lookback)
    roll_high = pd.Series(high_1w).rolling(window=50, min_periods=50).max().values
    roll_low = pd.Series(low_1w).rolling(window=50, min_periods=50).min().values
    
    # Avoid look-ahead: use previous period's values
    prev_high = np.roll(roll_high, 1)
    prev_low = np.roll(roll_low, 1)
    prev_high[0] = roll_high[0]
    prev_low[0] = roll_low[0]
    
    # Calculate Fibonacci levels
    diff = prev_high - prev_low
    fib_382 = prev_high - 0.382 * diff  # 38.2% retracement
    fib_618 = prev_high - 0.618 * diff  # 61.8% retracement
    
    # Align Fib levels to 12h timeframe
    fib_382_aligned = align_htf_to_ltf(prices, df_1w, fib_382)
    fib_618_aligned = align_htf_to_ltf(prices, df_1w, fib_618)
    
    # 12h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(ema200_aligned[i]) or np.isnan(fib_382_aligned[i]) or
            np.isnan(fib_618_aligned[i]) or np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema200_val = ema200_aligned[i]
        fib_382_val = fib_382_aligned[i]
        fib_618_val = fib_618_aligned[i]
        vol_ok = vol_filter[i]
        
        if position == 0:
            # Long: price near 61.8% Fib in uptrend (price > EMA200) with volume
            if price > ema200_val and abs(price - fib_618_val) < 0.01 * price and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price near 38.2% Fib in downtrend (price < EMA200) with volume
            elif price < ema200_val and abs(price - fib_382_val) < 0.01 * price and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 38.2% Fib or trend changes
            if price < fib_382_val or price < ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 61.8% Fib or trend changes
            if price > fib_618_val or price > ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1w_Fibonacci_1d_EMA200_TrendFilter_Volume_v1"
timeframe = "12h"
leverage = 1.0