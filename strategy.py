#!/usr/bin/env python3
"""
12h_1d_1w_camarilla_volume_v3
Hypothesis: Camarilla pivot levels from 1d combined with 1w trend filter and volume confirmation.
In bull markets: buy near S3/S4 with 1w uptrend. In bear markets: sell near R3/R4 with 1w downtrend.
Volume confirms institutional interest at pivot levels. Designed for low-frequency, high-conviction trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_camarilla_volume_v3"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r4 = close_1d + range_1d * 1.500
    r3 = close_1d + range_1d * 1.250
    r2 = close_1d + range_1d * 1.166
    r1 = close_1d + range_1d * 1.083
    s1 = close_1d - range_1d * 1.083
    s2 = close_1d - range_1d * 1.166
    s3 = close_1d - range_1d * 1.250
    s4 = close_1d - range_1d * 1.500
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Simple trend: price above/below 20-period EMA
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_1w_up = close_1w > ema_20_1w
    trend_1w_down = close_1w < ema_20_1w
    
    # Forward fill trends
    trend_1w_up_ffilled = pd.Series(trend_1w_up).ffill().values
    trend_1w_down_ffilled = pd.Series(trend_1w_down).ffill().values
    
    # Align 1d Camarilla levels to 12h
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Align 1w trend to 12h
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up_ffilled)
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down_ffilled)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches R3 or 1w trend turns down
            if (close[i] >= r3_aligned[i]) or trend_1w_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: price reaches S3 or 1w trend turns up
            if (close[i] <= s3_aligned[i]) or trend_1w_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: price at S3/S4 + 1w uptrend + volume
            if ((close[i] <= s3_aligned[i]) or (close[i] <= s4_aligned[i])) and \
               trend_1w_up_aligned[i] and volume_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price at R3/R4 + 1w downtrend + volume
            elif ((close[i] >= r3_aligned[i]) or (close[i] >= r4_aligned[i])) and \
                 trend_1w_down_aligned[i] and volume_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals