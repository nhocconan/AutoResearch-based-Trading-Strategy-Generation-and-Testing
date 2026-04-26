#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_VolumeSpike_HTFTrend_EMA_Cross
Hypothesis: Camarilla R3/S3 breakouts with volume spike (>2.0x 20-bar MA) and 1d EMA34/EMA89 cross trend filter. 
Uses tighter breakout levels (R3/S3) for stronger momentum confirmation. Volume spike confirms institutional interest. 
1d EMA34/EMA89 cross ensures trading with higher timeframe trend direction to reduce whipsaws in choppy markets. 
Fixed position sizing (0.25) to minimize fee churn. Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Previous 1d bar's OHLC for Camarilla levels (R3/S3 = stronger breakout levels)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla levels: R3, S3 (stronger breakout levels)
    rng = high_1d - low_1d
    camarilla_r3 = close_1d_vals + (rng * 1.1 / 4)   # R3 level
    camarilla_s3 = close_1d_vals - (rng * 1.1 / 4)   # S3 level
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d EMA34 and EMA89 for trend filter (EMA34 > EMA89 = uptrend, < = downtrend)
    ema_34_1d = pd.Series(close_1d_vals).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_1d = pd.Series(close_1d_vals).ewm(span=89, adjust=False, min_periods=89).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_89_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_89_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average (dynamic threshold)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Fixed position size to reduce churn (0.25 = 25% of capital)
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (89 for EMA, 20 for vol)
    start_idx = 89
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(ema_89_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        ema_34_val = ema_34_1d_aligned[i]
        ema_89_val = ema_89_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine 1d trend: EMA34 > EMA89 = uptrend, EMA34 < EMA89 = downtrend
        uptrend = ema_34_val > ema_89_val
        downtrend = ema_34_val < ema_89_val
        
        # Entry conditions: breakout of Camarilla R3/S3 with volume spike AND 1d EMA trend filter
        long_entry = (close_val > camarilla_r3_val) and vol_spike and uptrend
        short_entry = (close_val < camarilla_s3_val) and vol_spike and downtrend
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = fixed_size
                position = 1
            elif short_entry:
                signals[i] = -fixed_size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on mean reversion to midpoint (Camarilla center) OR trend change
            mid_point = (camarilla_r3_val + camarilla_s3_val) / 2
            trend_change = not uptrend  # Exit if trend turns down
            if close_val < mid_point or trend_change:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = fixed_size
        elif position == -1:
            # Short - exit on mean reversion to midpoint (Camarilla center) OR trend change
            mid_point = (camarilla_r3_val + camarilla_s3_val) / 2
            trend_change = not downtrend  # Exit if trend turns up
            if close_val > mid_point or trend_change:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -fixed_size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_VolumeSpike_HTFTrend_EMA_Cross"
timeframe = "4h"
leverage = 1.0