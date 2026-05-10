#!/usr/bin/env python3
"""
6H_ElderRay_BullBearPower_1dTrend_Filter
Hypothesis: Elder Ray (Bull Power/Bear Power) measures bull/bear strength relative to EMA13. 
Combine with 1d EMA50 trend filter and volume spike to capture strong momentum moves in both bull and bear markets.
Uses discrete position sizing (0.25) to limit churn. Designed for low trade frequency (<30/year) to minimize fee drag.
"""

name = "6H_ElderRay_BullBearPower_1dTrend_Filter"
timeframe = "6h"
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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA 50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume filter: volume > 2.0x 20-period average (tight to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 13)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        is_uptrend = close[i] > ema_50_1d_aligned[i]
        is_downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: Bull Power > 0 (strength) + volume spike + 1d uptrend
            if bull_power[i] > 0 and volume[i] > vol_threshold[i] and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power < 0 (weakness) + volume spike + 1d downtrend
            elif bear_power[i] < 0 and volume[i] > vol_threshold[i] and is_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power turns negative (loss of strength)
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power turns positive (loss of weakness)
            if bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals