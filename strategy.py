# 12h_DailyPivot_Breakout_Volume_TrendFilter_Strict_v2
# Hypothesis: Daily pivot levels (R1/S1) act as key support/resistance on 12h timeframe. 
# Breakouts with volume and trend filters capture institutional moves. Works in bull/bear by following momentum.
# Tightened filters to reduce trade frequency: increased volume threshold, added ATR filter for volatility regime.
# Timeframe: 12h, uses 1d for pivots. Target: 25-50 trades/year to avoid fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align daily pivot levels to 12h timeframe (use previous day's levels)
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume filter: current volume > 3.0 * 24-period average (12h bars)
    volume_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # ATR filter: only trade when volatility is above average (avoid chop)
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    atr_ma50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr > atr_ma50  # Trade only when volatility is above average
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need enough data for ATR MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_12h[i]) or np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or
            np.isnan(volume_ma24[i]) or np.isnan(volatility_filter[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (3.0 * volume_ma24[i])
        
        # Combined filter: volume AND volatility
        combined_filter = volume_filter and volatility_filter[i]
        
        if position == 0:
            # Long breakout: price breaks above R1 with filters
            if close[i] > r1_12h[i] and combined_filter:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with filters
            elif close[i] < s1_12h[i] and combined_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below S1
            if close[i] < s1_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above R1
            if close[i] > r1_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_DailyPivot_Breakout_Volume_TrendFilter_Strict_v2"
timeframe = "12h"
leverage = 1.0