#!/usr/bin/env python3
name = "6h_WilliamsAlligator_ElderRay_1wTrend"
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
    
    # === 1w Data for trend filter (Alligator) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === 1d Data for Elder Ray (Bull/Bear Power) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === Williams Alligator on 1w (Jaw: SMA13, Teeth: SMA8, Lips: SMA5) ===
    jaw_1w = pd.Series(close_1w).rolling(window=13, min_periods=13).mean().values
    teeth_1w = pd.Series(close_1w).rolling(window=8, min_periods=8).mean().values
    lips_1w = pd.Series(close_1w).rolling(window=5, min_periods=5).mean().values
    
    jaw_1w_aligned = align_htf_to_ltf(prices, df_1w, jaw_1w)
    teeth_1w_aligned = align_htf_to_ltf(prices, df_1w, teeth_1w)
    lips_1w_aligned = align_htf_to_ltf(prices, df_1w, lips_1w)
    
    # === Elder Ray on 1d: Bull Power = High - EMA13, Bear Power = Low - EMA13 ===
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # === Volume spike detection (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 13, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_1w_aligned[i]) or 
            np.isnan(teeth_1w_aligned[i]) or
            np.isnan(lips_1w_aligned[i]) or
            np.isnan(bull_power_1d_aligned[i]) or
            np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        alligator_bull = lips_1w_aligned[i] > teeth_1w_aligned[i] > jaw_1w_aligned[i]
        alligator_bear = lips_1w_aligned[i] < teeth_1w_aligned[i] < jaw_1w_aligned[i]
        
        if position == 0:
            # Long: Alligator bullish + Bull Power positive + volume spike
            if (alligator_bull and 
                bull_power_1d_aligned[i] > 0 and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish + Bear Power negative + volume spike
            elif (alligator_bear and 
                  bear_power_1d_aligned[i] < 0 and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns bearish OR Bull Power turns negative
            if alligator_bear or bull_power_1d_aligned[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns bullish OR Bear Power turns positive
            if alligator_bull or bear_power_1d_aligned[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals