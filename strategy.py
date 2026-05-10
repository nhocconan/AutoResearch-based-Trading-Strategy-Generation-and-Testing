#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeS
Hypothesis: Uses Camarilla pivot levels (R1/S1) from daily timeframe for breakout entries,
filtered by 1d EMA34 trend direction and volume confirmation. Designed for low trade
frequency (20-40/year) to minimize fee fatigue while capturing institutional breakout
levels that work in both bull and bear markets due to EMA trend filter and volume
validation.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeS"
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
    
    # 1d data for Camarilla pivots and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (R1, S1) from prior day
    # Formula: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels for each day
    camarilla_range = (high_1d - low_1d) * 1.1 / 12.0
    r1 = close_1d + camarilla_range  # Resistance 1
    s1 = close_1d - camarilla_range  # Support 1
    
    # Align to 4h timeframe (values apply to the FOLLOWING day)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need EMA34 to be valid
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or \
           np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.8x average
        volume_confirm = volume[i] > 1.8 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: price breaks above R1 with volume, above EMA34 (uptrend)
            if close[i] > r1_aligned[i] and volume_confirm and close[i] > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume, below EMA34 (downtrend)
            elif close[i] < s1_aligned[i] and volume_confirm and close[i] < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 or closes below EMA34
            if close[i] < s1_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R1 or closes above EMA34
            if close[i] > r1_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals