#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d Camarilla pivot levels + volume confirmation
# Long when price breaks above 6h Donchian high AND price > 1d Camarilla H3 + volume > 1.5x average
# Short when price breaks below 6h Donchian low AND price < 1d Camarilla L3 + volume > 1.5x average
# Exit when price crosses back inside Donchian channel or volume drops below average
# Works in bull markets via breakout continuation and bear markets via breakdown continuation
# Targets 50-150 total trades over 4 years with strict entry conditions

name = "6h_donchian_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Donchian Channel (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d Camarilla Pivot Levels
    df_1d = get_htf_data(prices, '1d')
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels: H4 = Close + 1.5*(High-Low), H3 = Close + 1.1*(High-Low), etc.
    daily_range = daily_high - daily_low
    camarilla_h3 = daily_close + 1.1 * daily_range
    camarilla_l3 = daily_close - 1.1 * daily_range
    camarilla_h4 = daily_close + 1.5 * daily_range
    camarilla_l4 = daily_close - 1.5 * daily_range
    
    # Align daily Camarilla levels to 6h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses back inside Donchian channel OR volume drops below average
        if position == 1:  # long position
            if close[i] < highest_high[i] or volume[i] < volume_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] > lowest_low[i] or volume[i] < volume_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with Camarilla filter and volume confirmation
            # Long: price breaks above Donchian high AND above H3 + volume confirmation
            if (close[i] > highest_high[i] and 
                close[i] > h3_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND below L3 + volume confirmation
            elif (close[i] < lowest_low[i] and 
                  close[i] < l3_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals