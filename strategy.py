#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla R1/S1 level breakouts with 1d EMA(50) trend filter and volume confirmation provide consistent edge in both bull and bear markets.
# Uses Camarilla pivot levels from 1d for structure, EMA(50) on 1d for trend alignment, and volume > 1.8x average for confirmation.
# Designed for low trade frequency (target: 20-40/year) with discrete sizing (0.25) to minimize fee drag.
# Camarilla levels are calculated from prior 1d high-low range: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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
    
    # Volume confirmation: > 1.8x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = 1.8 * vol_ma
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 and S1 from prior 1d bar
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1d trend filter (EMA50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d close for trend determination (also aligned)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(vol_ma[i]) or \
           np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or \
           np.isnan(ema_50_1d_aligned[i]) or np.isnan(close_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        is_uptrend = close_1d_aligned[i] > ema_50_1d_aligned[i]
        is_downtrend = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above Camarilla R1, volume confirmation, in uptrend
            if close[i] > camarilla_r1_aligned[i] and volume[i] > vol_threshold[i] and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Camarilla S1, volume confirmation, in downtrend
            elif close[i] < camarilla_s1_aligned[i] and volume[i] > vol_threshold[i] and is_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below Camarilla S1 or trend turns down
            if close[i] < camarilla_s1_aligned[i] or is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above Camarilla R1 or trend turns up
            if close[i] > camarilla_r1_aligned[i] or is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals