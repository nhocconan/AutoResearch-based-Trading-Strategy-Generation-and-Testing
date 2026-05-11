#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_Dyn_v1
Hypothesis: Camarilla pivot levels (R1/S1) from daily high/low/close act as intraday support/resistance.
Breakouts above R1 or below S1 with volume confirmation and 1d EMA34 trend filter capture momentum.
Volume spike (volume > 1.5 * 20-period average) filters breakouts. Works in bull (breakouts continue)
and bear (breakouts fade) via EMA34 trend filter. Target: 20-50 trades/year on 4h timeframe.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_Dyn_v1"
timeframe = "4h"
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
    
    # === 1D Data for Camarilla Pivots and EMA34 ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R1, S1
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_width = 1.1 * (high_1d - low_1d) / 12.0
    r1_1d = close_1d + camarilla_width
    s1_1d = close_1d - camarilla_width
    
    # Align Camarilla levels to 4h
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Volume Spike Filter (20-period average) ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: breakout above R1 with volume spike and above EMA34 (uptrend)
            if (close[i] > r1_1d_aligned[i]) and volume_spike[i] and (ema34_1d_aligned[i] < close[i]):
                signals[i] = 0.25
                position = 1
            # Short: breakdown below S1 with volume spike and below EMA34 (downtrend)
            elif (close[i] < s1_1d_aligned[i]) and volume_spike[i] and (ema34_1d_aligned[i] > close[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below EMA34 (trend change) or breaks below S1 (reversal)
            if (ema34_1d_aligned[i] > close[i]) or (close[i] < s1_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price crosses above EMA34 (trend change) or breaks above R1 (reversal)
            if (ema34_1d_aligned[i] < close[i]) or (close[i] > r1_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals