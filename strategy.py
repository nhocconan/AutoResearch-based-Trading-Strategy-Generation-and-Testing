#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
Hypothesis: 1h Camarilla R1/S1 breakout filtered by 4h trend and volume > 1.3x average.
Uses 4h for directional bias (trend) and 1h for precise entry timing to avoid false breaks.
Works in bull via breakout continuation and in bear via mean-reversion off extremes.
Target: 80-120 total trades over 4 years (20-30/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 4h EMA(20) for trend
    close_4h = df_4h['close'].values
    ema_period = 20
    ema_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= ema_period:
        ema_4h[ema_period-1] = np.mean(close_4h[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, len(close_4h)):
            ema_4h[i] = (close_4h[i] * multiplier) + (ema_4h[i-1] * (1 - multiplier))
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1h data for Camarilla calculation (same timeframe as prices)
    df_1h = prices[['high', 'low', 'close']].copy()
    
    # Calculate Camarilla R1 and S1 for each 1h bar
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    camarilla_r1 = np.zeros(len(close_1h))
    camarilla_s1 = np.zeros(len(close_1h))
    for i in range(len(close_1h)):
        if high_1h[i] == low_1h[i]:
            camarilla_r1[i] = close_1h[i]
            camarilla_s1[i] = close_1h[i]
        else:
            camarilla_r1[i] = close_1h[i] + (high_1h[i] - low_1h[i]) * 1.1 / 12
            camarilla_s1[i] = close_1h[i] - (high_1h[i] - low_1h[i]) * 1.1 / 12
    
    # Volume confirmation: 20-period MA
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.20  # 20% position size
    
    # Warmup: need volume MA (20)
    start_idx = vol_ma_period
    
    for i in range(start_idx, n):
        if np.isnan(vol_ma[i]) or np.isnan(ema_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filter: price above/below 4h EMA(20)
        uptrend = price > ema_4h_aligned[i]
        downtrend = price < ema_4h_aligned[i]
        
        # Volume confirmation: > 1.3x average volume
        volume_confirmation = vol_ratio > 1.3
        
        if position == 0:
            # Long entry: price breaks above R1 in uptrend with volume
            if price > camarilla_r1[i] and uptrend and volume_confirmation:
                signals[i] = size
                position = 1
            # Short entry: price breaks below S1 in downtrend with volume
            elif price < camarilla_s1[i] and downtrend and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns below S1 or trend reverses
            if price < camarilla_s1[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price returns above R1 or trend reverses
            if price > camarilla_r1[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0