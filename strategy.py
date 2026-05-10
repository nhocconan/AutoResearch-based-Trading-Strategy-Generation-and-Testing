#!/usr/bin/env python3
# 4H_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike
# Hypothesis: Uses Camarilla pivot levels R1/S1 on 12h chart for breakout signals,
# filtered by 12h EMA50 trend and volume spike (1.5x 20-period average) to avoid false breakouts.
# Works in bull/bear markets by taking breakouts in direction of 12h trend.
# Targets 20-50 trades per year on 4h timeframe with position size 0.25.

name = "4H_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 12h data for Camarilla pivots and EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for 12h
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    r1_12h = df_12h['close'] + (df_12h['high'] - df_12h['low']) * 1.1 / 12
    s1_12h = df_12h['close'] - (df_12h['high'] - df_12h['low']) * 1.1 / 12
    
    # Calculate 12h EMA(50) for trend
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h indicators to 4h
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h.values)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h.values)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike: 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for EMA50
    
    for i in range(start_idx, n):
        if np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or np.isnan(ema_50_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R1 with uptrend and volume spike
            if (close[i] > r1_12h_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below S1 with downtrend and volume spike
            elif (close[i] < s1_12h_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below EMA50 or breaks below S1 (reversal)
            if (close[i] < ema_50_12h_aligned[i] or 
                close[i] < s1_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above EMA50 or breaks above R1 (reversal)
            if (close[i] > ema_50_12h_aligned[i] or 
                close[i] > r1_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals