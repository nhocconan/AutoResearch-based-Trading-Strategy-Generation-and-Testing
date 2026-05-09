#!/usr/bin/env python3
# 4H_1D_Camarilla_R1_S1_Breakout_1dTrend_Volume_v2
# Hypothesis: Use tighter volume confirmation (2x avg) and require trend confirmation from both 1d and 1w to reduce trades.
# Only trade when price breaks Camarilla R1/S1 with strong volume, aligned trend on 1d and 1w, and avoid choppy markets.
# Target: 15-30 trades/year per symbol (60-120 total over 4 years) to avoid fee drag.

name = "4H_1D_Camarilla_R1_S1_Breakout_1dTrend_Volume_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w data
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    # 1d calculations for Camarilla and trend
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels from previous 1d
    typical_price = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    camarilla_r1 = close_1d + (range_1d * 1.1 / 12)
    camarilla_s1 = close_1d - (range_1d * 1.1 / 12)
    
    # 1d trend: EMA(34)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_1d = close_1d > ema_34_1d
    
    # 1w trend: EMA(20)
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_up_1w = close_1w > ema_20_1w
    
    # Volume confirmation: current volume > 2x 20-period average (tighter)
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 2.0)
    
    # Align all to 4h
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or \
           np.isnan(trend_up_1d_aligned[i]) or np.isnan(trend_up_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Camarilla R1 + 1d uptrend + 1w uptrend + volume confirmation
            if (close[i] > camarilla_r1_aligned[i] and 
                trend_up_1d_aligned[i] and 
                trend_up_1w_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Camarilla S1 + 1d downtrend + 1w downtrend + volume confirmation
            elif (close[i] < camarilla_s1_aligned[i] and 
                  not trend_up_1d_aligned[i] and 
                  not trend_up_1w_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Camarilla S1 or trend deteriorates
            if close[i] < camarilla_s1_aligned[i] or not trend_up_1d_aligned[i] or not trend_up_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Camarilla R1 or trend improves
            if close[i] > camarilla_r1_aligned[i] or trend_up_1d_aligned[i] or trend_up_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals