#!/usr/bin/env python3
# 1D_Camarilla_R1S3_Breakout_WeeklyTrend_Volume
# Hypothesis: Breakouts at key weekly Camarilla levels (R1 for longs, S3 for shorts) on 1d timeframe with volume confirmation and weekly trend alignment capture momentum moves while avoiding whipsaws. Uses strict entry conditions to limit trades and reduce fee drag. Works in bull/bear by following weekly trend direction.

name = "1D_Camarilla_R1S3_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
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
    # S1 = close - (range * 1.0833)
    # S3 = close - (range * 1.2500)
    # R1 = close + (range * 1.0833)
    # R3 = close + (range * 1.2500)
    s1 = close_1w - (range_1w * 1.08333)
    s3 = close_1w - (range_1w * 1.25000)
    r1 = close_1w + (range_1w * 1.08333)
    r3 = close_1w + (range_1w * 1.25000)
    
    # Align to 1d timeframe (wait for weekly bar to close)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    
    # Weekly trend filter: EMA 34
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume filter: volume > 2.0x 50-period average (strict to limit trades)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_threshold = vol_ma * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(s1_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        is_uptrend = close[i] > ema_34_1w_aligned[i]
        is_downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long entry: Price breaks above R1 + volume confirmation + weekly uptrend
            if (close[i] > r1_aligned[i] and 
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
            # Long exit: Price crosses below S1 (opposite side)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price crosses above R1 (opposite side)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals