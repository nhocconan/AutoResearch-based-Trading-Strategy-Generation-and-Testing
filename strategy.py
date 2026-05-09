#!/usr/bin/env python3
name = "1d_WeeklyTrend_DailyBreakout_12hVolume"
timeframe = "1d"
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Get 12h data for volume filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    
    # Calculate 50-period EMA on 1w close
    ema50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        ema50_1w[49] = np.mean(close_1w[0:50])
        for i in range(50, len(close_1w)):
            ema50_1w[i] = (close_1w[i] * 2 + ema50_1w[i-1] * 48) / 50
    
    # Align 1w EMA50 to 1d timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 20-period SMA of volume on 12h
    vol_ma_12h = np.full_like(volume_12h, np.nan)
    if len(volume_12h) >= 20:
        vol_ma_12h[19] = np.mean(volume_12h[0:20])
        for i in range(20, len(volume_12h)):
            vol_ma_12h[i] = (vol_ma_12h[i-1] * 19 + volume_12h[i]) / 20
    
    # Align 12h volume MA to 1d timeframe
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Calculate volume ratio: current volume vs 12h volume MA
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma_12h_aligned)) & (vol_ma_12h_aligned != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma_12h_aligned[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need 1w EMA50 and 12h volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market conditions
        trend_up = close[i] > ema50_1w_aligned[i]
        volume_surge = volume_ratio[i] > 1.5
        
        if position == 0:
            # Enter long: Uptrend + volume surge
            if trend_up and volume_surge:
                signals[i] = 0.25
                position = 1
            # Enter short: Downtrend + volume surge
            elif not trend_up and volume_surge:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Trend turns down OR volume normalizes
            if not trend_up or volume_ratio[i] < 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Trend turns up OR volume normalizes
            if trend_up or volume_ratio[i] < 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals