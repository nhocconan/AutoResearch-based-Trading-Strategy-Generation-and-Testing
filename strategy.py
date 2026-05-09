#!/usr/bin/env python3
name = "4H_Camarilla_Pivot_1DTrend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for Camarilla pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily EMA34 for trend filter
    if len(close_1d) >= 34:
        ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    else:
        ema34_1d = np.full_like(close_1d, np.nan)
    
    # Align daily EMA34 to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla pivot levels from previous day
    # Classic Camarilla: H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    # We use H3/L3 for entry: H3 = C + 1.1*(H-L)/4, L3 = C - 1.1*(H-L)/4
    camarilla_h3 = np.full_like(close_1d, np.nan)
    camarilla_l3 = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        if i >= 1:  # Need previous day's data
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            prev_close = close_1d[i-1]
            range_ = prev_high - prev_low
            camarilla_h3[i] = prev_close + 1.1 * range_ / 4
            camarilla_l3[i] = prev_close - 1.1 * range_ / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Get 1h data for volume confirmation (shorter lookback for responsiveness)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    volume_1h = df_1h['volume'].values
    
    # Calculate 1h volume EMA20
    if len(volume_1h) >= 20:
        vol_ema20_1h = pd.Series(volume_1h).ewm(span=20, adjust=False, min_periods=20).mean().values
    else:
        vol_ema20_1h = np.full_like(volume_1h, np.nan)
    
    # Align 1h volume EMA20 to 4h timeframe
    vol_ema20_1h_aligned = align_htf_to_ltf(prices, df_1h, vol_ema20_1h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = max(1, 34, 20)  # Need Camarilla (1 day), EMA34, volume EMA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ema20_1h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market conditions
        # Uptrend: price above daily EMA34
        uptrend = close[i] > ema34_1d_aligned[i]
        # Downtrend: price below daily EMA34
        downtrend = close[i] < ema34_1d_aligned[i]
        # Volume surge: current volume > 1.5x 1h volume EMA20
        volume_surge = volume[i] > vol_ema20_1h_aligned[i] * 1.5
        
        if position == 0:
            # Enter long: Uptrend + price touches/breaks above Camarilla H3 + volume surge
            if uptrend and close[i] >= camarilla_h3_aligned[i] and volume_surge:
                signals[i] = 0.25
                position = 1
            # Enter short: Downtrend + price touches/breaks below Camarilla L3 + volume surge
            elif downtrend and close[i] <= camarilla_l3_aligned[i] and volume_surge:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Trend turns down OR price touches/breaks below Camarilla L3
            if not uptrend or close[i] <= camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Trend turns up OR price touches/breaks above Camarilla H3
            if not downtrend or close[i] >= camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals