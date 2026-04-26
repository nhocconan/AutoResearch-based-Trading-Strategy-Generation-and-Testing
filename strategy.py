#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeFilter
Hypothesis: On daily timeframe, Camarilla R1/S1 breakouts with volume confirmation (top 30%) and 1-week EMA50 trend filter. Only trade breakouts aligned with weekly trend to avoid counter-trend whipsaws. Uses discrete position sizing (0.25) to limit trades. Target: 15-25 trades/year. Designed to work in both bull (trend continuation) and bear (mean reversion to weekly mean) markets via trend filter and midpoint exit.
"""

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
    
    # Load 1w data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Previous 1w bar's OHLC for Camarilla levels (R1/S1 = standard breakout levels)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_vals = df_1w['close'].values
    
    # Calculate Camarilla levels: R1, S1 (standard breakout levels)
    rng = high_1w - low_1w
    camarilla_r1 = close_1w_vals + (rng * 1.1 / 12)   # R1 level
    camarilla_s1 = close_1w_vals - (rng * 1.1 / 12)   # S1 level
    
    # Align Camarilla levels to 1d timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # 1w EMA50 for trend filter
    close_1w_series = pd.Series(close_1w_vals)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: volume > 70th percentile of 50-period lookback (avoid low-volume breakouts)
    vol_series = pd.Series(volume)
    vol_percentile_70 = vol_series.rolling(window=50, min_periods=50).quantile(0.70).values
    volume_filter = volume > vol_percentile_70
    
    # Discrete position size to control trade frequency and fees
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (50 for EMA and volume percentile)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_percentile_70[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        ema_50_val = ema_50_1w_aligned[i]
        vol_filt = volume_filter[i]
        size = fixed_size
        
        # Entry conditions: breakout of Camarilla R1/S1 with volume filter AND aligned with 1w EMA50 trend
        long_entry = (close_val > camarilla_r1_val) and vol_filt and (close_val > ema_50_val)
        short_entry = (close_val < camarilla_s1_val) and vol_filt and (close_val < ema_50_val)
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on mean reversion to Camarilla midpoint (center)
            mid_point = (camarilla_r1_val + camarilla_s1_val) / 2
            if close_val < mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on mean reversion to Camarilla midpoint (center)
            mid_point = (camarilla_r1_val + camarilla_s1_val) / 2
            if close_val > mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0