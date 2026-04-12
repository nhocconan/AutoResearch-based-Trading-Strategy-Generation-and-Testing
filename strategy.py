#!/usr/bin/env python3
"""
4h_1d_Camarilla_Breakout_SignalStrength_v1
Hypothesis: Use daily Camarilla pivot levels with volume-weighted signal strength and trend filter.
Only trade when price breaks H3/L3 with strong volume confirmation (volume > 2x average) and price is beyond 1% of the level.
Long in uptrend (price > 50 EMA), short in downtrend (price < 50 EMA).
Designed for low trade frequency (<40/year) with high conviction signals.
Works in bull markets (continuation breaks) and bear markets (mean reversion from extremes).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Breakout_SignalStrength_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY DATA FOR CAMARILLA PIVOTS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    typical_price = (high_1d + low_1d + close_1d) / 3
    pivot = typical_price
    range_1d = high_1d - low_1d
    
    h3 = pivot + (range_1d * 1.1 / 4)
    l3 = pivot - (range_1d * 1.1 / 4)
    h4 = pivot + (range_1d * 1.1 / 2)
    l4 = pivot - (range_1d * 1.1 / 2)
    
    # Align to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # === TREND FILTER: 50 EMA ON 4H CHART ===
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === VOLUME FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(ema50[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine trend
        uptrend = close[i] > ema50[i]
        downtrend = close[i] < ema50[i]
        
        # Volume strength (must be significantly above average)
        strong_volume = volume[i] > (vol_ma[i] * 2.0)
        
        # Price must be beyond the level by at least 1% to avoid false breakouts
        level_buffer = 0.01
        
        # Long: price breaks H3 with strong volume in uptrend
        long_signal = (close[i] > h3_aligned[i] * (1 + level_buffer) and 
                      uptrend and 
                      strong_volume)
        
        # Short: price breaks L3 with strong volume in downtrend
        short_signal = (close[i] < l3_aligned[i] * (1 - level_buffer) and 
                       downtrend and 
                       strong_volume)
        
        # Exit: opposite H3/L3 level or trend reversal
        exit_long = (position == 1 and 
                    (close[i] < l3_aligned[i] or not uptrend))
        exit_short = (position == -1 and 
                     (close[i] > h3_aligned[i] or not downtrend))
        
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