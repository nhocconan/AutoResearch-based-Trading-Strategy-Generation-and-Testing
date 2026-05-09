#!/usr/bin/env python3
# 4H_Trend_Angle_Breakout_1dVol
# Hypothesis: Buy when price breaks above 4h Donchian(20) high with rising price angle (linear regression slope > 0) and volume >1.5x 20-bar average; sell when breaks below Donchian(20) low with falling price angle (slope < 0) and volume confirmation. Uses 1d volume for regime filter to avoid low-volatility chop. Designed for 20-40 trades/year on 4h timeframe.

name = "4H_Trend_Angle_Breakout_1dVol"
timeframe = "4h"
leverage = 1.0

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
    
    # Donchian channels (20-period)
    donchian_high = np.full_like(high, np.nan)
    donchian_low = np.full_like(low, np.nan)
    if len(high) >= 20:
        for i in range(20-1, len(high)):
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
    
    # Price angle: linear regression slope over 5 periods
    price_slope = np.full_like(close, np.nan)
    if len(close) >= 5:
        for i in range(4, len(close)):
            y = close[i-4:i+1]
            x = np.arange(5)
            slope = np.polyfit(x, y, 1)[0]
            price_slope[i] = slope
    
    # Get 1d volume for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full_like(vol_1d, np.nan)
    if len(vol_1d) >= 20:
        for i in range(19, len(vol_1d)):
            vol_ma_1d[i] = np.mean(vol_1d[i-19:i+1])
    
    vol_ratio_1d = np.full_like(vol_1d, np.nan)
    valid_vol = (~np.isnan(vol_ma_1d)) & (vol_ma_1d != 0)
    vol_ratio_1d[valid_vol] = vol_1d[valid_vol] / vol_ma_1d[valid_vol]
    
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # Volume ratio for entry confirmation (4h)
    vol_ma_4h = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma_4h[i] = np.mean(volume[i-19:i+1])
    
    vol_ratio_4h = np.full_like(volume, np.nan)
    valid_vol_4h = (~np.isnan(vol_ma_4h)) & (vol_ma_4h != 0)
    vol_ratio_4h[valid_vol_4h] = volume[valid_vol_4h] / vol_ma_4h[valid_vol_4h]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 4, 19)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(price_slope[i]) or np.isnan(vol_ratio_4h[i]) or \
           np.isnan(vol_ratio_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Break above Donchian high + rising price angle + volume confirmation (both TFs)
            if close[i] > donchian_high[i] and price_slope[i] > 0 and vol_ratio_4h[i] > 1.5 and vol_ratio_1d_aligned[i] > 1.2:
                signals[i] = 0.25
                position = 1
            # Enter short: Break below Donchian low + falling price angle + volume confirmation
            elif close[i] < donchian_low[i] and price_slope[i] < 0 and vol_ratio_4h[i] > 1.5 and vol_ratio_1d_aligned[i] > 1.2:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Break below Donchian low or price angle turns negative
            if close[i] < donchian_low[i] or price_slope[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Break above Donchian high or price angle turns positive
            if close[i] > donchian_high[i] or price_slope[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals