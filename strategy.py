#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime
Hypothesis: 12h Camarilla R1/S1 breakouts with 1d EMA34 trend filter and volume spike regime (top 20% volume). Uses wider breakout levels (R1/S1) for more frequent but valid signals. 1d EMA34 ensures alignment with daily trend to avoid counter-trend whipsaws. Volume spike ensures trades occur during high participation, reducing false breakouts. Fixed position size 0.25 to control trade frequency and fees. Target: 12-25 trades/year.
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
    
    # Load 1d data ONCE before loop for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous 1d bar's OHLC for Camarilla levels (R1/S1 = wider breakout levels)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla levels: R1, S1 (wider breakout levels for more signals)
    rng = high_1d - low_1d
    camarilla_r1 = close_1d_vals + (rng * 1.1 / 2)   # R1 level
    camarilla_s1 = close_1d_vals - (rng * 1.1 / 2)   # S1 level
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1d EMA34 for trend filter (long only above EMA, short only below EMA)
    ema_34_1d = pd.Series(close_1d_vals).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume regime: volume > 80th percentile of 50-period lookback (volume spike days only)
    vol_series = pd.Series(volume)
    vol_percentile_80 = vol_series.rolling(window=50, min_periods=50).quantile(0.80).values
    volume_regime = volume > vol_percentile_80
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (50 for volume percentile, 34 for EMA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_percentile_80[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_regime = volume_regime[i]
        size = fixed_size
        
        # Entry conditions: breakout of Camarilla R1/S1 with volume regime AND 1d EMA34 trend filter
        long_entry = (close_val > camarilla_r1_val) and vol_regime and (close_val > ema_34_val)
        short_entry = (close_val < camarilla_s1_val) and vol_regime and (close_val < ema_34_val)
        
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

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime"
timeframe = "12h"
leverage = 1.0