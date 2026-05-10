#!/usr/bin/env python3
# 4h_Camarilla_R1S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot R1/S1 breakout with 1-day trend filter and volume confirmation.
# Long when price breaks above R1 in uptrend (close > EMA34_1d) with volume > 1.5x average.
# Short when price breaks below S1 in downtrend (close < EMA34_1d) with volume > 1.5x average.
# Works in bull/bear markets by aligning with higher timeframe trend.
# Targets 20-50 trades per year on 4h timeframe with position size 0.25.

name = "4h_Camarilla_R1S1_Breakout_1dTrend_Volume"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend direction
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 40-period average volume for volume filter
    vol_avg = pd.Series(volume).rolling(window=40, min_periods=40).mean().values
    
    # Calculate daily Camarilla levels (R1, S1) from previous day
    # Need at least 2 days of data
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values  # Shifted by 1 day
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla R1 and S1 for previous day
    # R1 = Close + (High - Low) * 1.12
    # S1 = Close - (High - Low) * 1.12
    r1 = prev_close + (prev_high - prev_low) * 1.12
    s1 = prev_close - (prev_high - prev_low) * 1.12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for volume average
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_avg[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 1d
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter: current volume > 1.5x average
        volume_filter = volume[i] > 1.5 * vol_avg[i]
        
        if position == 0:
            # Long entry: price breaks above R1 AND uptrend AND volume confirmation
            if close[i] > r1_aligned[i] and uptrend and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 AND downtrend AND volume confirmation
            elif close[i] < s1_aligned[i] and downtrend and volume_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 OR trend turns down
            if close[i] < s1_aligned[i] or downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R1 OR trend turns up
            if close[i] > r1_aligned[i] or uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals