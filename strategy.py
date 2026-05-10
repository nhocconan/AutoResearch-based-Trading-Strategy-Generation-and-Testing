#!/usr/bin/env python3
# 12H_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike
# Hypothesis: Camarilla R3/S3 levels from weekly timeframe act as strong institutional support/resistance.
# Breakouts with volume > 2x average and weekly EMA50 trend alignment capture strong momentum moves.
# Weekly trend filter ensures we only trade in the direction of the higher timeframe trend, working in both bull and bear markets.
# Targets 12-37 trades per year on 12h timeframe.

name = "12H_Camarilla_R3S3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
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
    
    # Weekly Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous weekly bar
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla formulas: range = high - low
    range_1w = high_1w - low_1w
    # S3 = close - (range * 1.2500)
    # R3 = close + (range * 1.2500)
    s3 = close_1w - (range_1w * 1.25000)
    r3 = close_1w + (range_1w * 1.25000)
    
    # Align to 12h timeframe (wait for weekly bar to close)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    
    # Weekly trend filter: EMA 50
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        is_uptrend = close[i] > ema_50_1w_aligned[i]
        is_downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long entry: Price breaks above R3 + volume confirmation + weekly uptrend
            if (close[i] > r3_aligned[i] and 
                volume[i] > vol_threshold[i] and 
                is_uptrend):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below S3 + volume confirmation + weekly downtrend
            elif (close[i] < s3_aligned[i] and 
                  volume[i] > vol_threshold[i] and 
                  is_downtrend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price crosses below S3 (opposite side)
            if close[i] < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price crosses above R3 (opposite side)
            if close[i] > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals