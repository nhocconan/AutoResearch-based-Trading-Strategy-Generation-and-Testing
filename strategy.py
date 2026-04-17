#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR and range calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily true range and ATR
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily range (high - low) and its 14-day average
    daily_range = high_1d - low_1d
    range_avg = pd.Series(daily_range).rolling(window=14, min_periods=14).mean().values
    
    # Align ATR and range average to 12h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    range_avg_aligned = align_htf_to_ltf(prices, df_1d, range_avg)
    
    # Volatility filter: current daily range > 1.2 * average range (avoid low volatility)
    # Need to align the condition: we want to know if today's range is expanded
    range_expanded = daily_range > (1.2 * range_avg)
    range_expanded_aligned = align_htf_to_ltf(prices, df_1d, range_expanded.astype(float))
    
    # Calculate 12-period high and low for breakout levels (using 12h data directly)
    high_roll = pd.Series(high).rolling(window=12, min_periods=12).max().values
    low_roll = pd.Series(low).rolling(window=12, min_periods=12).min().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need sufficient data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_aligned[i]) or 
            np.isnan(range_avg_aligned[i]) or 
            np.isnan(range_expanded_aligned[i]) or
            np.isnan(high_roll[i]) or 
            np.isnan(low_roll[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: trade only when volatility is expanding
        volatility_filter = range_expanded_aligned[i] > 0.5  # True when range > 1.2*avg
        
        if position == 0:
            # Long: price breaks above 12-period high with expanding volatility
            if close[i] > high_roll[i] and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12-period low with expanding volatility
            elif close[i] < low_roll[i] and volatility_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below 12-period low or volatility contracts
            if close[i] < low_roll[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above 12-period high or volatility contracts
            if close[i] > high_roll[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_VolatilityBreakout_12Period_v1"
timeframe = "12h"
leverage = 1.0