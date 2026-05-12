#!/usr/bin/env python3
name = "1d_Camarilla_R3_S4_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
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
    
    # === 1w Data for Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # === Calculate Weekly EMA34 for Trend Filter ===
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === Calculate Daily Camarilla Levels (R3, S4) from Previous Day ===
    # Need daily high/low for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R3 = C + (H-L)*1.1/4, S4 = C - (H-L)*1.1/4
    rng = high_1d - low_1d
    R3 = close_1d + rng * 1.1 / 4
    S4 = close_1d - rng * 1.1 / 4
    
    # Align 1w EMA34 to 1d timeframe (weekly trend available after weekly close)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    # Camarilla levels are based on previous day, so align with 1-day delay
    R3_1d = align_htf_to_ltf(prices, df_1d, R3)
    S4_1d = align_htf_to_ltf(prices, df_1d, S4)
    
    # === Volume Spike Detection (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(R3_1d[i]) or
            np.isnan(S4_1d[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 + above 1w EMA34 + volume spike
            if (close[i] > R3_1d[i] and 
                close[i] > ema34_1w_aligned[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S4 + below 1w EMA34 + volume spike
            elif (close[i] < S4_1d[i] and 
                  close[i] < ema34_1w_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below S4 (reversal) or below 1w EMA34
            if close[i] < S4_1d[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above R3 (reversal) or above 1w EMA34
            if close[i] > R3_1d[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals