#!/usr/bin/env python3
# 4H_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
# Hypothesis: Uses daily Camarilla pivot levels (R1/S1) with breakout confirmation.
# Enters long when price breaks above R1 with volume spike and 1-day uptrend (close > EMA50).
# Enters short when price breaks below S1 with volume spike and 1-day downtrend (close < EMA50).
# Exits when price returns to the 1-day EMA50 level.
# Uses 1-day EMA50 for trend filter and volume > 1.5x 20-period average for confirmation.
# Targets 20-50 trades per year on 4h timeframe with position size 0.25.

name = "4H_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
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
    
    # Get 1d data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1-day Camarilla pivot levels (R1, S1)
    # Camarilla formulas: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    r1_1d = df_1d['close'].values + (df_1d['high'].values - df_1d['low'].values) * 1.1 / 12
    s1_1d = df_1d['close'].values - (df_1d['high'].values - df_1d['low'].values) * 1.1 / 12
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate volume spike: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above R1 with volume spike in uptrend
            if (close[i] > r1_1d_aligned[i] and 
                volume_spike[i] and 
                price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 with volume spike in downtrend
            elif (close[i] < s1_1d_aligned[i] and 
                  volume_spike[i] and 
                  price_below_ema):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to or below 1-day EMA50
            if close[i] <= ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to or above 1-day EMA50
            if close[i] >= ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals