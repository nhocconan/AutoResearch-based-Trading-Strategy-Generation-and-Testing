#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_VolumeFilter
Hypothesis: Camarilla pivot levels (R1/S1) from 1d act as strong support/resistance. 
Price breaking above R1 or below S1 with volume confirmation (volume > 1.5x average) 
triggers entries. Only trade in direction of 1w trend (EMA50) to avoid counter-trend 
whipsaws. Works in bull/bear by aligning with higher timeframe trend.
Timeframe: 12h (lower frequency reduces trade count, mitigating fee drag).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla levels and 1w EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 and S1 (based on previous day)
    # R1 = Close + 1.1*(High - Low)/12
    # S1 = Close - 1.1*(High - Low)/12
    camarilla_r1 = np.full_like(close_1d, np.nan)
    camarilla_s1 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        if not np.isnan(high_1d[i-1]) and not np.isnan(low_1d[i-1]) and not np.isnan(close_1d[i-1]):
            camarilla_r1[i] = close_1d[i-1] + 1.1 * (high_1d[i-1] - low_1d[i-1]) / 12
            camarilla_s1[i] = close_1d[i-1] - 1.1 * (high_1d[i-1] - low_1d[i-1]) / 12
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1w EMA trend filter (weekly trend direction)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 50:
        close_1w = df_1w['close'].values
        ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
        ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    else:
        ema_50_1w_aligned = np.full(n, np.nan)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        volume_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
        volume_confirmed = volume_ratio > 1.5
        
        # Trend filter: only trade in direction of 1w EMA50
        trend_up = price > ema_50_1w_aligned[i]
        trend_down = price < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume confirmation + uptrend
            if price > r1 and volume_confirmed and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume confirmation + downtrend
            elif price < s1 and volume_confirmed and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price drops below S1 or trend reverses
            if price < s1 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above R1 or trend reverses
            if price > r1 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_VolumeFilter"
timeframe = "12h"
leverage = 1.0