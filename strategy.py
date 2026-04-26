#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike
Hypothesis: On 1h timeframe, trade Camarilla R1/S1 breakouts from prior 4h candle with volume spike (top 30%) only when aligned with 4h EMA50 trend. Use session filter (08-20 UTC) to avoid low-liquidity hours. Fixed size 0.20 to limit trades. Target: 15-30 trades/year per symbol. Works in bull via trend continuation, in bear via mean-reversion exits at midpoint.
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
    
    # Load 4h data ONCE before loop for HTF filters
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Previous 4h bar's OHLC for Camarilla levels (R1/S1 = standard breakout levels)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_vals = df_4h['close'].values
    
    # Calculate Camarilla levels: R1, S1 (standard breakout levels)
    rng = high_4h - low_4h
    camarilla_r1 = close_4h_vals + (rng * 1.1 / 12)   # R1 level
    camarilla_s1 = close_4h_vals - (rng * 1.1 / 12)   # S1 level
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # 4h EMA50 for trend filter
    close_4h_series = pd.Series(close_4h_vals)
    ema_50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume spike: volume > 70th percentile of 20-period lookback (volume confirmation)
    vol_series = pd.Series(volume)
    vol_percentile_70 = vol_series.rolling(window=20, min_periods=20).quantile(0.70).values
    volume_spike = volume > vol_percentile_70
    
    # Session filter: 08-20 UTC (avoid low-liquidity hours)
    hours = prices.index.hour  # open_time is already datetime64[ms], .hour works
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Fixed position size to control trade frequency
    fixed_size = 0.20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (50 for EMA, 20 for volume percentile)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_percentile_70[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        ema_50_val = ema_50_4h_aligned[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        # Entry conditions: breakout of Camarilla R1/S1 with volume spike AND aligned with 4h EMA50 trend
        long_entry = (close_val > camarilla_r1_val) and vol_spike and (close_val > ema_50_val)
        short_entry = (close_val < camarilla_s1_val) and vol_spike and (close_val < ema_50_val)
        
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
            # Long - exit on mean reversion to midpoint (Camarilla center)
            mid_point = (camarilla_r1_val + camarilla_s1_val) / 2
            if close_val < mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on mean reversion to midpoint (Camarilla center)
            mid_point = (camarilla_r1_val + camarilla_s1_val) / 2
            if close_val > mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0