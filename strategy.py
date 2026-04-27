#!/usr/bin/env python3
"""
4h_382_Retracement_Retest_Trend_Confirm
Hypothesis: Enter on 38.2% Fibonacci retracement of the prior day's range in the direction of the 1d EMA34 trend. This captures pullback entries during trending moves, which are higher-probability than breakouts. Works in bull (buy dips) and bear (sell rallies) markets. Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 35:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Fibonacci levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Fibonacci levels from previous 1d range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    range_1d = high_1d - low_1d
    # 38.2% retracement levels
    retracement_long = low_1d + range_1d * 0.382  # for uptrend: buy pullback to 38.2%
    retracement_short = high_1d - range_1d * 0.382  # for downtrend: sell rally to 38.2%
    
    # Align to 4h timeframe (previous day's levels available at open)
    retracement_long_aligned = align_htf_to_ltf(prices, df_1d, retracement_long)
    retracement_short_aligned = align_htf_to_ltf(prices, df_1d, retracement_short)
    
    # Volume filter: require volume > 1.5x 20-period average to avoid low-volatility false signals
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 20  # need 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(retracement_long_aligned[i]) or 
            np.isnan(retracement_short_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price pulls back to 38.2% level in uptrend with volume confirmation
            if (close[i] >= retracement_long_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price rallies to 38.2% level in downtrend with volume confirmation
            elif (close[i] <= retracement_short_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below 61.8% level or trend fails
            fib_618_long = low_1d + range_1d * 0.618
            fib_618_long_aligned = align_htf_to_ltf(prices, df_1d, fib_618_long)
            if (close[i] < fib_618_long_aligned[i] or 
                close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above 61.8% level or trend fails
            fib_618_short = high_1d - range_1d * 0.618
            fib_618_short_aligned = align_htf_to_ltf(prices, df_1d, fib_618_short)
            if (close[i] > fib_618_short_aligned[i] or 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_382_Retracement_Retest_Trend_Confirm"
timeframe = "4h"
leverage = 1.0