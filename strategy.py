#!/usr/bin/env python3
name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "6h"
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
    
    # === 1D DATA FOR CAMARILLA PIVOTS AND TREND ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # Pivot = (H + L + C) / 3
    # R3 = C + (H - L) * 1.1 / 2
    # S3 = C - (H - L) * 1.1 / 2
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    r3_1d = close_1d + (high_1d - low_1d) * 1.1 / 2
    s3_1d = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (use previous day's values)
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # 1D EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_6h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === VOLUME CONFIRMATION (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)  # Strong volume spike for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(ema34_1d_6h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R3 with volume + price above 1d EMA34 (uptrend)
            if (close[i] > r3_6h[i] and 
                ema34_1d_6h[i] > 0 and  # Ensure EMA is valid
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 with volume + price below 1d EMA34 (downtrend)
            elif (close[i] < s3_6h[i] and 
                  ema34_1d_6h[i] > 0 and  # Ensure EMA is valid
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price falls back below R3 or trend weakens
            if close[i] < r3_6h[i] or ema34_1d_6h[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above S3 or trend weakens
            if close[i] > s3_6h[i] or ema34_1d_6h[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals