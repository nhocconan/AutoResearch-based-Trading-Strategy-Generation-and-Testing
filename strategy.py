#!/usr/bin/env python3
"""
4h_12h_Camarilla_Trend_Breakout_v1
Hypothesis: Trade 12h Camarilla H3/L3 breakouts with volume confirmation and 12h trend filter.
Long when price breaks above H3 in uptrend (price > 12h EMA200), short when breaks below L3 in downtrend.
Exit on trend reversal or opposite level touch. Designed for low trade frequency (<30/year) with high conviction.
Works in bull markets (continuation breaks) and bear markets (mean reversion from extremes).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Camarilla_Trend_Breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12H DATA FOR TREND AND CAMARILLA ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 200:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h EMA200 for trend
    close_12h_series = pd.Series(close_12h)
    ema200_12h = close_12h_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 12h Camarilla levels from previous 12h bar
    typical_price_12h = (high_12h + low_12h + close_12h) / 3
    pivot_12h = typical_price_12h
    range_12h = high_12h - low_12h
    
    h3_12h = pivot_12h + (range_12h * 1.1 / 4)
    l3_12h = pivot_12h - (range_12h * 1.1 / 4)
    h4_12h = pivot_12h + (range_12h * 1.1 / 2)
    l4_12h = pivot_12h - (range_12h * 1.1 / 2)
    
    # Align 12h indicators to 4h timeframe
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    h3_12h_aligned = align_htf_to_ltf(prices, df_12h, h3_12h)
    l3_12h_aligned = align_htf_to_ltf(prices, df_12h, l3_12h)
    h4_12h_aligned = align_htf_to_ltf(prices, df_12h, h4_12h)
    l4_12h_aligned = align_htf_to_ltf(prices, df_12h, l4_12h)
    
    # === VOLUME CONFIRMATION (4H) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if not ready
        if (np.isnan(ema200_12h_aligned[i]) or np.isnan(h3_12h_aligned[i]) or 
            np.isnan(l3_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 12h trend
        uptrend = close[i] > ema200_12h_aligned[i]
        downtrend = close[i] < ema200_12h_aligned[i]
        
        # Volume strength (must be significantly above average)
        strong_volume = volume[i] > (vol_ma[i] * 2.0)
        
        # Price must be beyond the level by at least 1% to avoid false breakouts
        level_buffer = 0.01
        
        # Long: price breaks above H3_12h in uptrend with strong volume
        long_signal = (close[i] > h3_12h_aligned[i] * (1 + level_buffer) and 
                      uptrend and 
                      strong_volume)
        
        # Short: price breaks below L3_12h in downtrend with strong volume
        short_signal = (close[i] < l3_12h_aligned[i] * (1 - level_buffer) and 
                       downtrend and 
                       strong_volume)
        
        # Exit: opposite H3/L3 level or trend reversal
        exit_long = (position == 1 and 
                    (close[i] < l3_12h_aligned[i] or not uptrend))
        exit_short = (position == -1 and 
                     (close[i] > h3_12h_aligned[i] or not downtrend))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals