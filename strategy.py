#!/usr/bin/env python3
"""
1d_1D_Camarilla_R3_S3_Breakout_1wTrend_Volume
Hypothesis: Use 1w EMA trend filter with Camarilla R3/S3 breakout on 1d, requiring volume confirmation.
Camarilla levels provide high-probability reversal/breakout points. Weekly EMA filter ensures trading with weekly trend.
Volume filter avoids false breakouts. Target: 15-25 trades/year, works in bull/bear via trend filter.
"""

name = "1d_1D_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data for price and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla calculation
    camarilla_range = high_prev - low_prev
    camarilla_r3 = close_prev + 1.1 * camarilla_range * 1.1 / 4  # Close + 1.1*(HL)/4 * 1.1
    camarilla_s3 = close_prev - 1.1 * camarilla_range * 1.1 / 4  # Close - 1.1*(HL)/4 * 1.1
    
    # Align Camarilla levels to 1d timeframe (already 1d, no alignment needed)
    camarilla_r3_aligned = camarilla_r3
    camarilla_s3_aligned = camarilla_s3
    
    # Get 1d price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.8x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA34 and previous day data
    start_idx = 35
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1w trend: price vs EMA34
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: uptrend AND price breaks above Camarilla R3 with volume
            if uptrend and high[i] > camarilla_r3_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: downtrend AND price breaks below Camarilla S3 with volume
            elif downtrend and low[i] < camarilla_s3_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Camarilla S3 OR trend changes to downtrend
            if low[i] < camarilla_s3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Camarilla R3 OR trend changes to uptrend
            if high[i] > camarilla_r3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals