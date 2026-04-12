#!/usr/bin/env python3
# 6h_1d_elder_ray_reversal_with_volume
# Hypothesis: Elder Ray (Bull/Bear Power) with volume confirmation on 6h timeframe.
# Uses daily EMA(13) as trend filter and Elder Ray to detect exhaustion/reversal.
# Works in bull/bear by only taking counter-trend moves when Elder Ray diverges from price.
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.

name = "6h_1d_elder_ray_reversal_with_volume"
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
    
    # Get daily data for EMA and Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily EMA(13) for trend filter
    ema13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema13
    bear_power = low_1d - ema13
    
    # Align daily indicators to 6h timeframe
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(ema13_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: price below EMA (downtrend) but Bull Power rising (bulls gaining strength)
        if (close[i] < ema13_aligned[i] and 
            bull_power_aligned[i] > bull_power_aligned[i-1] and
            bull_power_aligned[i-1] <= bull_power_aligned[i-2] and  # turning up from recent low
            vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: price above EMA (uptrend) but Bear Power falling (bears gaining strength)
        elif (close[i] > ema13_aligned[i] and 
              bear_power_aligned[i] < bear_power_aligned[i-1] and
              bear_power_aligned[i-1] >= bear_power_aligned[i-2] and  # turning down from recent high
              vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: Elder Ray confirms trend resumption or reverse signal
        elif position == 1 and (bull_power_aligned[i] < 0 or close[i] > ema13_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bear_power_aligned[i] > 0 or close[i] < ema13_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals