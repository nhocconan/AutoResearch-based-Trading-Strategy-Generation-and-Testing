#!/usr/bin/env python3
"""
12h_1w_1d_Camarilla_Pivot_Breakout_Trend
Hypothesis: On 12h chart, price breaks above/below weekly Camarilla R4/S4 levels with volume confirmation and EMA trend filter. Uses weekly pivot for stronger institutional levels. Works in bull/bear by trading breakouts in direction of weekly EMA trend. Target: 12-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivots and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    close_prev = np.roll(close_1w, 1)
    close_prev[0] = close_1w[0]
    
    range_1w = high_1w - low_1w
    
    # Resistance levels
    R1 = close_prev + (range_1w * 1.0833 / 12)
    R2 = close_prev + (range_1w * 1.1666 / 6)
    R3 = close_prev + (range_1w * 1.2500 / 4)
    R4 = close_prev + (range_1w * 1.5000 / 2)
    
    # Support levels
    S1 = close_prev - (range_1w * 1.0833 / 12)
    S2 = close_prev - (range_1w * 1.1666 / 6)
    S3 = close_prev - (range_1w * 1.2500 / 4)
    S4 = close_prev - (range_1w * 1.5000 / 2)
    
    # Weekly EMA for trend filter
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly levels to 12h
    R4_aligned = align_htf_to_ltf(prices, df_1w, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1w, S4)
    ema_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # Daily volume confirmation
    df_1d = get_htf_data(prices, '1d')
    vol_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_20_aligned = align_htf_to_ltf(prices, df_1d, vol_20)
    volume_expansion = volume > (vol_20_aligned * 2.0)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or 
            np.isnan(ema_aligned[i]) or np.isnan(volume_expansion[i])):
            continue
            
        # Trend filter: price above/below weekly EMA
        uptrend = close[i] > ema_aligned[i]
        downtrend = close[i] < ema_aligned[i]
        
        # Long: break above R4 in uptrend with volume
        if uptrend and close[i] > R4_aligned[i] and volume_expansion[i]:
            if position != 1:
                position = 1
                signals[i] = position_size
            else:
                signals[i] = position_size
        # Short: break below S4 in downtrend with volume
        elif downtrend and close[i] < S4_aligned[i] and volume_expansion[i]:
            if position != -1:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = -position_size
        else:
            # Hold or flatten
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1w_1d_Camarilla_Pivot_Breakout_Trend"
timeframe = "12h"
leverage = 1.0