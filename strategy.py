#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_VolumeConfirm
Hypothesis: Daily Donchian(20) breakout with 1-week EMA50 trend filter and volume confirmation (>1.8x median).
Enters long when price breaks above upper Donchian with volume confirmation and bullish 1w trend (close > EMA50).
Enters short when price breaks below lower Donchian with volume confirmation and bearish 1w trend (close < EMA50).
Exits on opposite Donchian breakout or when price crosses the 20-period EMA.
Uses discrete position sizing (0.25) to minimize churn. Target: 30-100 trades over 4 years.
Works in both bull and bear markets by following 1w trend filter and volatility-based volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels for 1d (based on previous 20 days)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Previous 20-day high/low for Donchian (to avoid look-ahead)
    h_20 = pd.Series(h_1d).rolling(window=20, min_periods=20).max().shift(1).values
    l_20 = pd.Series(l_1d).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align to 1d primary timeframe
    upper_donchian = align_htf_to_ltf(prices, df_1d, h_20)
    lower_donchian = align_htf_to_ltf(prices, df_1d, l_20)
    
    # Volume confirmation: volume > 1.8x 60-period median (stricter for daily)
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=60, min_periods=60).median().values
    volume_confirm = volume > (1.8 * vol_median)
    
    # Load 1w data for HTF trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 20-period EMA for exit (optional mean reversion touch)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 60-period volume median, 50-period weekly EMA, 20-period EMA)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or 
            np.isnan(vol_median[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: price breaks above upper Donchian + volume confirmation + bullish 1w trend
        if close[i] > upper_donchian[i] and volume_confirm[i] and close[i] > ema_50_1w_aligned[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below lower Donchian + volume confirmation + bearish 1w trend
        elif close[i] < lower_donchian[i] and volume_confirm[i] and close[i] < ema_50_1w_aligned[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit: opposite Donchian breakout or price crosses 20-period EMA (mean reversion)
        elif position == 1 and (close[i] < lower_donchian[i] or close[i] < ema_20[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > upper_donchian[i] or close[i] > ema_20[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0