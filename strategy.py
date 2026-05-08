#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h price action combined with weekly pivot levels and volume confirmation.
# Long when price breaks above weekly R2 pivot level AND volume > 1.5x 20-period average.
# Short when price breaks below weekly S2 pivot level AND volume > 1.5x 20-period average.
# Exit when price returns to weekly pivot (PP) level.
# Weekly pivots capture institutional levels that hold across multiple timeframes.
# Volume confirmation filters false breakouts. Weekly timeframe provides structural bias.
# Designed for 60-120 total trades over 4 years (15-30/year) to minimize fee drag.

name = "6h_WeeklyPivot_R2S2_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    # Based on previous week's OHLC
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    # Pivot point and support/resistance levels
    PP = (prev_high + prev_low + prev_close) / 3.0
    R1 = 2 * PP - prev_low
    S1 = 2 * PP - prev_high
    R2 = PP + (prev_high - prev_low)
    S2 = PP - (prev_high - prev_low)
    R3 = prev_high + 2 * (PP - prev_low)
    S3 = prev_low - 2 * (prev_high - PP)
    
    # Align weekly pivot levels to 6h timeframe
    PP_aligned = align_htf_to_ltf(prices, df_1w, PP)
    R2_aligned = align_htf_to_ltf(prices, df_1w, R2)
    S2_aligned = align_htf_to_ltf(prices, df_1w, S2)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 2)  # Sufficient warmup for volume MA and weekly data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(PP_aligned[i]) or np.isnan(R2_aligned[i]) or 
            np.isnan(S2_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above weekly R2, volume filter
            long_cond = (close[i] > R2_aligned[i]) and volume_filter[i]
            # Short conditions: price breaks below weekly S2, volume filter
            short_cond = (close[i] < S2_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to weekly pivot (PP) level
            if close[i] <= PP_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to weekly pivot (PP) level
            if close[i] >= PP_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals