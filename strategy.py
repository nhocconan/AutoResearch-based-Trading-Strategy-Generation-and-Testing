#!/usr/bin/env python3
name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for previous 1d (R1, S1)
    # Pivot = (H + L + C) / 3
    # R1 = Pivot + (H - L) * 1.1 / 12
    # S1 = Pivot - (H - L) * 1.1 / 12
    H = df_1d['high'].values
    L = df_1d['low'].values
    C = df_1d['close'].values
    
    pivot = (H + L + C) / 3.0
    r1 = pivot + (H - L) * 1.1 / 12.0
    s1 = pivot - (H - L) * 1.1 / 12.0
    
    # Align 1d Camarilla levels to 12h timeframe (use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # 1d volume average for volume spike detection (20-period)
    vol_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_20_aligned = align_htf_to_ltf(prices, df_1d, vol_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # sufficient warmup for EMA and volume
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(vol_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike condition: current 12h volume > 2.0 * 1d average volume
        # Need to approximate 12h volume from 1d - we'll use current volume vs 1d average scaled
        # Since we don't have 12h volume directly, we'll use price action as primary signal
        # and use 1d volume spike as confirmation
        vol_spike = volume[i] > (vol_20_aligned[i] * 2.0) if not np.isnan(vol_20_aligned[i]) else False
        
        if position == 0:
            # Long: price breaks above R1 + uptrend + volume spike
            if close[i] > r1_aligned[i] and close[i] > ema_34_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + downtrend + volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema_34_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price crosses back through the pivot level
            pivot_aligned = (r1_aligned[i] + s1_aligned[i]) / 2.0  # approximate pivot
            if position == 1:
                if close[i] < pivot_aligned:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > pivot_aligned:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals