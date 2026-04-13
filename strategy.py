#!/usr/bin/env python3
"""
6h_12h_1d_Camarilla_Pivot_Breakout
Hypothesis: Uses daily Camarilla pivot levels (calculated from prior day's range) to identify key support/resistance.
Enters on 6h break of R4/S4 levels with volume confirmation, using 12h trend filter to avoid counter-trend trades.
Works in bull markets (breakouts continue) and bear markets (breakdowns continue) by trading momentum after volatility expansion.
Target: 15-35 trades/year on 6h (60-140 total over 4 years).
"""

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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day's range
    # R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    range_1d = high_1d - low_1d
    camarilla_r4 = close_1d + range_1d * 1.1 / 2
    camarilla_r3 = close_1d + range_1d * 1.1 / 4
    camarilla_s3 = close_1d - range_1d * 1.1 / 4
    camarilla_s4 = close_1d - range_1d * 1.1 / 2
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Simple trend: price above/below 20-period EMA
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_up = close_12h > ema_20_12h
    trend_down = close_12h < ema_20_12h
    
    # Align all signals to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    trend_up_aligned = align_htf_to_ltf(prices, df_12h, trend_up)
    trend_down_aligned = align_htf_to_ltf(prices, df_12h, trend_down)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if not in session or data not ready
        if not session_mask[i] or \
           np.isnan(camarilla_r4_aligned[i]) or \
           np.isnan(camarilla_r3_aligned[i]) or \
           np.isnan(camarilla_s3_aligned[i]) or \
           np.isnan(camarilla_s4_aligned[i]) or \
           np.isnan(trend_up_aligned[i]) or \
           np.isnan(trend_down_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Entry conditions: price breaks S4/R4 with volume confirmation and trend alignment
        vol_ma_20_6h = pd.Series(volume[max(0, i-19):i+1]).mean() if i >= 20 else 0
        volume_expansion = volume[i] > (vol_ma_20_6h * 1.5) if i >= 20 else False
        
        # Long: price breaks above R4 with volume and uptrend
        if (close[i] > camarilla_r4_aligned[i] and 
            volume_expansion and 
            trend_up_aligned[i]):
            if position != 1:
                position = 1
                signals[i] = position_size
            else:
                signals[i] = position_size
        # Short: price breaks below S4 with volume and downtrend
        elif (close[i] < camarilla_s4_aligned[i] and 
              volume_expansion and 
              trend_down_aligned[i]):
            if position != -1:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = -position_size
        # Hold current position
        elif position == 1:
            signals[i] = position_size
        elif position == -1:
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_12h_1d_Camarilla_Pivot_Breakout"
timeframe = "6h"
leverage = 1.0