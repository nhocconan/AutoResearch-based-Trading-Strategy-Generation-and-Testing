#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
Hypothesis: Combine 4h trend direction with 1h Camarilla breakout for precise entries.
Use 4h EMA(50) for trend filter and 1h Camarilla R1/S1 levels for breakout entries.
Volume confirmation >1.5x average filters false breakouts.
Target: 15-35 trades/year by requiring trend alignment + volume + breakout.
Works in bull/bear: follows 4h trend, captures breakouts in direction of trend.
"""

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
    
    # Get 4h data for trend filter and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_period = 50
    ema_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= ema_period:
        ema_4h[ema_period - 1] = np.mean(close_4h[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, len(close_4h)):
            ema_4h[i] = (close_4h[i] * multiplier) + (ema_4h[i-1] * (1 - multiplier))
    
    # Calculate 4h volume moving average (20-period) for volume filter
    volume_4h = df_4h['volume'].values
    vol_ma_period = 20
    vol_ma_4h = np.full(len(volume_4h), np.nan)
    for i in range(vol_ma_period, len(volume_4h)):
        vol_ma_4h[i] = np.mean(volume_4h[i-vol_ma_period:i+1])
    
    # Calculate daily OHLC for Camarilla levels (use previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_R1 = np.full(len(close_1d), np.nan)
    camarilla_S1 = np.full(len(close_1d), np.nan)
    camarilla_PP = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        hl_range = high_1d[i-1] - low_1d[i-1]
        camarilla_PP[i] = (high_1d[i-1] + low_1d[i-1] + close_1d[i-1]) / 3
        camarilla_R1[i] = close_1d[i-1] + (hl_range * 1.1 / 12)
        camarilla_S1[i] = close_1d[i-1] - (hl_range * 1.1 / 12)
    
    # Align 4h indicators to 1h timeframe
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    camarilla_PP_aligned = align_htf_to_ltf(prices, df_1d, camarilla_PP)
    
    # 1h volume confirmation (20-period average)
    vol_ma_1h_period = 20
    vol_ma_1h = np.full(n, np.nan)
    for i in range(vol_ma_1h_period, n):
        vol_ma_1h[i] = np.mean(volume[i-vol_ma_1h_period:i])
    
    signals = np.zeros(n)
    position = 0
    
    # Session filter: 08-20 UTC (active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need all indicators
    start_idx = max(1, 50, 20, 20)  # Camarilla needs 1 day, EMA(50), vol MA(20) both timeframes
    
    for i in range(start_idx, n):
        # Session filter: only trade 08-20 UTC
        if hours[i] < 8 or hours[i] > 20:
            signals[i] = 0.0
            continue
            
        if (np.isnan(camarilla_R1_aligned[i]) or
            np.isnan(camarilla_S1_aligned[i]) or
            np.isnan(camarilla_PP_aligned[i]) or
            np.isnan(ema_4h_aligned[i]) or
            np.isnan(vol_ma_4h_aligned[i]) or
            np.isnan(vol_ma_1h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_1h[i] if vol_ma_1h[i] > 0 else 0
        
        # Trend filter: price above/below 4h EMA50
        uptrend = price > ema_4h_aligned[i]
        downtrend = price < ema_4h_aligned[i]
        
        # Volume confirmation: > 1.5x average 1h volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long: price breaks above Camarilla R1 in uptrend with volume
            if uptrend and volume_confirmation and price > camarilla_R1_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S1 in downtrend with volume
            elif downtrend and volume_confirmation and price < camarilla_S1_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price touches Camarilla PP or trend reverses
            if price <= camarilla_PP_aligned[i] or price < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20  # Maintain position
        elif position == -1:
            # Short exit: price touches Camarilla PP or trend reverses
            if price >= camarilla_PP_aligned[i] or price > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20  # Maintain position
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0