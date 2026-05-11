#!/usr/bin/env python3
"""
6h_Aroon_Trend_Filter_1dTrend_VolumeSpike
Hypothesis: Use Aroon oscillator (25-period) to detect strong trends on 6h, filtered by 1d trend (EMA34) and volume spike (1.5x 20 EMA). Aroon > 50 indicates uptrend, < -50 downtrend. Designed for low trade frequency (~15-30/year) to avoid fee drag, works in bull/bear by aligning with daily trend.
"""

name = "6h_Aroon_Trend_Filter_1dTrend_VolumeSpike"
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
    
    # === Daily Trend Filter (EMA34) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_6h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Aroon Oscillator (25-period) on 6h ===
    aroon_period = 25
    aroon_up = np.full(n, np.nan)
    aroon_down = np.full(n, np.nan)
    
    for i in range(aroon_period - 1, n):
        period_high = high[i - aroon_period + 1:i + 1]
        period_low = low[i - aroon_period + 1:i + 1]
        
        # Find periods since highest high and lowest low
        high_idx = np.argmax(period_high)  # 0 to 24
        low_idx = np.argmin(period_low)    # 0 to 24
        
        aroon_up[i] = ((aroon_period - 1 - high_idx) / (aroon_period - 1)) * 100
        aroon_down[i] = ((aroon_period - 1 - low_idx) / (aroon_period - 1)) * 100
    
    aroon_osc = aroon_up - aroon_down  # -100 to +100
    
    # === Volume Filter (1.5x 20-period EMA on 6h) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers Aroon and daily EMA)
    start_idx = max(34, aroon_period - 1) + 5
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema34_6h[i]) or np.isnan(aroon_osc[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: strong uptrend (Aroon > 50) + uptrend on daily + volume spike
            if (aroon_osc[i] > 50 and 
                close[i] > ema34_6h[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short: strong downtrend (Aroon < -50) + downtrend on daily + volume spike
            elif (aroon_osc[i] < -50 and 
                  close[i] < ema34_6h[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend weakness (Aroon < 0) or trend reversal
            if aroon_osc[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: trend weakness (Aroon > 0) or trend reversal
            if aroon_osc[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals