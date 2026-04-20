#!/usr/bin/env python3
# 6h_Fibonacci_Pullback_BullBear_Trend
# Hypothesis: In trending markets, price pulls back to 61.8% Fibonacci retracement of recent swing before continuing.
# Uses 1d trend filter (EMA50) to determine trend direction, and 6h swing high/low for Fibonacci levels.
# Works in bull markets (long at pullbacks in uptrend) and bear markets (short at pullbacks in downtrend).
# Uses volume confirmation and ATR volatility filter to avoid false signals.
# Target: 12-37 trades/year (50-150 total over 4 years).

name = "6h_Fibonacci_Pullback_BullBear_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1h data for EMA50 trend filter (as proxy for 1d due to availability)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1h close
    ema50_1h = pd.Series(df_1h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1h_aligned = align_htf_to_ltf(prices, df_1h, ema50_1h)
    
    # Calculate 6h swing high/low for Fibonacci (using 20-period window)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate Fibonacci levels: 61.8% retracement
    diff = highest_20 - lowest_20
    fib_618_long = lowest_20 + 0.618 * diff  # For uptrend: pullback to 61.8% from low
    fib_618_short = highest_20 - 0.618 * diff  # For downtrend: pullback to 61.8% from high
    
    # Align Fibonacci levels (they are already 6h, but ensure alignment for safety)
    fib_618_long_aligned = align_htf_to_ltf(prices, prices, fib_618_long)  # self-align
    fib_618_short_aligned = align_htf_to_ltf(prices, prices, fib_618_short)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.3)
    
    # ATR filter: only trade when volatility is sufficient
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr > (atr_ma50 * 0.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1h_aligned[i]) or np.isnan(fib_618_long_aligned[i]) or
            np.isnan(fib_618_short_aligned[i]) or np.isnan(volume_filter[i]) or
            np.isnan(volatility_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Uptrend: price > EMA50, long at 61.8% pullback
            if close[i] > ema50_1h_aligned[i] and low[i] <= fib_618_long_aligned[i] and volume_filter[i] and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
            # Downtrend: price < EMA50, short at 61.8% pullback
            elif close[i] < ema50_1h_aligned[i] and high[i] >= fib_618_short_aligned[i] and volume_filter[i] and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long exit: price breaks below swing low or reverses to downtrend
            if close[i] < ema50_1h_aligned[i] or low[i] < lowest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short exit: price breaks above swing high or reverses to uptrend
            if close[i] > ema50_1h_aligned[i] or high[i] > highest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals