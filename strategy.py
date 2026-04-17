#!/usr/bin/env python3
"""
12h Prior Day High/Low Breakout with Volume Spike and Trend Filter
Long: Close breaks above prior day high AND volume > 2x 12h volume SMA(20) AND price > 1d EMA(50)
Short: Close breaks below prior day low AND volume > 2x 12h volume SMA(20) AND price < 1d EMA(50)
Exit: Close crosses back below prior day high (long) or above prior day low (short)
Targets 15-25 trades/year per symbol (60-100 total over 4 years)
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
    
    # Get 1d data for prior day levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Prior day high and low
    prior_high = df_1d['high'].values
    prior_low = df_1d['low'].values
    prior_high_aligned = align_htf_to_ltf(prices, df_1d, prior_high)
    prior_low_aligned = align_htf_to_ltf(prices, df_1d, prior_low)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h volume SMA(20) for volume filter
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(50, 20)  # EMA50 and SMA20 warmup
    
    for i in range(start_idx, n):
        if (np.isnan(prior_high_aligned[i]) or np.isnan(prior_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma[i]
        prior_high_val = prior_high_aligned[i]
        prior_low_val = prior_low_aligned[i]
        ema_1d_val = ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: break above prior day high + volume spike + price > 1d EMA50
            if price > prior_high_val and vol > 2.0 * vol_sma_val and price > ema_1d_val:
                signals[i] = 0.25
                position = 1
            # Short: break below prior day low + volume spike + price < 1d EMA50
            elif price < prior_low_val and vol > 2.0 * vol_sma_val and price < ema_1d_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below prior day high
            if price < prior_high_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above prior day low
            if price > prior_low_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Prior1D_HL_Breakout_VolumeSpike_TrendFilter"
timeframe = "12h"
leverage = 1.0