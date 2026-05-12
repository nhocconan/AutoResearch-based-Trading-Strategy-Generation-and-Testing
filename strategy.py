#!/usr/bin/env python3
name = "6h_ElderRay_BullPower_BearPower_1dTrend"
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
    
    # === 1D DATA FOR ELDER RAY AND TREND ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA13 for Elder Ray power calculation
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    # EMA50 for 1d trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high_1d - ema13_1d
    # Bear Power = Low - EMA13 (negative values indicate bearish pressure)
    bear_power = low_1d - ema13_1d
    
    # Align to 6h timeframe
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power)
    ema50_1d_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === VOLUME SPIKE (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_6h[i]) or 
            np.isnan(bear_power_6h[i]) or
            np.isnan(ema50_1d_6h[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 (strength) + price above 1d EMA50 (uptrend) + volume spike
            if (bull_power_6h[i] > 0 and 
                close[i] > ema50_1d_6h[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 (weakness) + price below 1d EMA50 (downtrend) + volume spike
            elif (bear_power_6h[i] < 0 and 
                  close[i] < ema50_1d_6h[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Bull Power turns negative OR price breaks below 1d EMA50
            if bull_power_6h[i] <= 0 or close[i] < ema50_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power turns positive OR price breaks above 1d EMA50
            if bear_power_6h[i] >= 0 or close[i] > ema50_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals