#!/usr/bin/env python3
"""
6h Elder Ray Power + 1d Trend Filter
Hypothesis: Elder Ray (bull/bear power) measures bullish/bearish momentum. 
Combined with 1d EMA trend filter and volume confirmation, it captures
sustained moves in both bull and bear markets while avoiding false signals
in ranging markets. Designed for low trade frequency (~20-30/year) to minimize
fee decay.
"""
name = "6h_ElderRay_Power_1dTrend_Volume"
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
    
    # === Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # === 1d EMA34 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Volume Spike (20) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 (bullish momentum) + price above 1d EMA34 + volume spike
            if (bull_power[i] > 0 and 
                close[i] > ema34_1d_aligned[i] and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 (bearish momentum) + price below 1d EMA34 + volume spike
            elif (bear_power[i] < 0 and 
                  close[i] < ema34_1d_aligned[i] and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Bear Power >= 0 (momentum fading) OR price below 1d EMA34
            if bear_power[i] >= 0 or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bull Power <= 0 (momentum fading) OR price above 1d EMA34
            if bull_power[i] <= 0 or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals