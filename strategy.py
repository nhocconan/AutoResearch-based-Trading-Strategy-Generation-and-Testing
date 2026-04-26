#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolRegime
Hypothesis: 1h Camarilla R1/S1 breakouts with 4h EMA50 trend filter and 1d volume spike regime. Uses wider R1/S1 levels for more frequent but still selective breakouts. 4h trend ensures directional bias, 1d volume spike confirms institutional participation. Fixed size 0.20 to limit trades. Target: 15-35 trades/year.
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
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data ONCE before loop for volume regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume spike: volume > 1.5 * 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > (1.5 * vol_ma_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # Calculate 1h Camarilla levels (R1/S1 = wider breakout levels)
    # Use 1h rolling window for Camarilla calculation (more responsive)
    high_1h = pd.Series(high).rolling(window=24, min_periods=24).max().values  # 24*1h = 1 day
    low_1h = pd.Series(low).rolling(window=24, min_periods=24).min().values
    close_1h = close  # current close for level calculation
    
    rng = high_1h - low_1h
    camarilla_r1 = close_1h + (rng * 1.1 / 6)   # R1 level
    camarilla_s1 = close_1h - (rng * 1.1 / 6)   # S1 level
    
    # Session filter: 08-20 UTC (reduces noise trades)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Fixed position size to control trade frequency
    fixed_size = 0.20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (50 for 4h EMA, 24 for Camarilla, 20 for 1d vol)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(camarilla_r1[i]) or 
            np.isnan(camarilla_s1[i])):
            signals[i] = 0.0
            continue
        
        # Skip outside session
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        camarilla_r1_val = camarilla_r1[i]
        camarilla_s1_val = camarilla_s1[i]
        trend_up = close_val > ema_50_4h_aligned[i]
        trend_down = close_val < ema_50_4h_aligned[i]
        vol_regime = vol_spike_1d_aligned[i] > 0.5  # boolean as float
        size = fixed_size
        
        # Entry conditions: breakout of Camarilla R1/S1 with 4h trend AND 1d volume spike
        long_entry = (close_val > camarilla_r1_val) and trend_up and vol_regime
        short_entry = (close_val < camarilla_s1_val) and trend_down and vol_regime
        
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
            # Long - exit on trend reversal or mean reversion to Camarilla center
            mid_point = (camarilla_r1_val + camarilla_s1_val) / 2
            trend_reversal = close_val < ema_50_4h_aligned[i]
            mean_reversion = close_val < mid_point
            if trend_reversal or mean_reversion:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal or mean reversion to Camarilla center
            mid_point = (camarilla_r1_val + camarilla_s1_val) / 2
            trend_reversal = close_val > ema_50_4h_aligned[i]
            mean_reversion = close_val > mid_point
            if trend_reversal or mean_reversion:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolRegime"
timeframe = "1h"
leverage = 1.0