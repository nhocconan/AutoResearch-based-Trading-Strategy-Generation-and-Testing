#!/usr/bin/env python3
"""
1h_4H_1D_Camarilla_R3_S3_Breakout_Trend_Volume
Hypothesis: Use 4h/1d trend filters with 1h entry timing to reduce noise. 
Camarilla R3/S3 breakouts on 1h with volume confirmation. 
Trend filters: price > 4h EMA50 (uptrend) or price < 4h EMA50 (downtrend). 
Volume filter: 1h volume > 1.5x 20-period EMA to avoid false breakouts.
Target: 15-35 trades/year per symbol with controlled risk via trend alignment.
Works in bull/bear via 4h trend filter. Low trade frequency minimizes fee drag.
"""

name = "1h_4H_1D_Camarilla_R3_S3_Breakout_Trend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data for trend filter and volume reference
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla calculation (previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h volume EMA20 for volume filter reference
    volume_4h = df_4h['volume'].values
    vol_ema20_4h = pd.Series(volume_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ema20_4h)
    
    # Calculate Camarilla levels from previous day (1d)
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla calculation: R3 = Close + 1.1*(HL)/4, S3 = Close - 1.1*(HL)/4
    camarilla_range = high_prev - low_prev
    camarilla_r3 = close_prev + 1.1 * camarilla_range / 4
    camarilla_s3 = close_prev - 1.1 * camarilla_range / 4
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 1h price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 4h volume EMA20 (adaptive to volatility)
    volume_filter = volume > vol_ema20_4h_aligned * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50 and previous day data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 4h trend: price vs EMA50
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # Long: uptrend AND price breaks above Camarilla R3 with volume
            if uptrend and high[i] > camarilla_r3_aligned[i] and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: downtrend AND price breaks below Camarilla S3 with volume
            elif downtrend and low[i] < camarilla_s3_aligned[i] and volume_filter[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below Camarilla S3 OR trend changes to downtrend
            if low[i] < camarilla_s3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above Camarilla R3 OR trend changes to uptrend
            if high[i] > camarilla_r3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals