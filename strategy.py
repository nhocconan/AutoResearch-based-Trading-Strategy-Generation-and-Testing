# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# 4H_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS
# Hypothesis: Trade 4h breakouts of daily Camarilla R1/S1 levels with 12h EMA50 trend filter and volume confirmation.
# Camarilla levels provide institutional-grade support/resistance; EMA50 on 12h filters trend direction to avoid counter-trend trades.
# Volume ensures breakout momentum. Designed for low frequency (20-50 trades/year) to survive both bull and bear markets.

name = "4H_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 12h EMA50 for trend filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === Daily Camarilla levels (R1, S1) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Use previous day's OHLC for current day's Camarilla
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    d_high_prev = np.roll(d_high, 1)
    d_low_prev = np.roll(d_low, 1)
    d_close_prev = np.roll(d_close, 1)
    d_high_prev[0] = np.nan
    d_low_prev[0] = np.nan
    d_close_prev[0] = np.nan
    
    # Camarilla calculation
    rng = d_high_prev - d_low_prev
    camarilla_r1 = d_close_prev + rng * 1.1 / 12
    camarilla_s1 = d_close_prev - rng * 1.1 / 12
    
    # Align daily levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === Volume confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA50
        trend_up = close[i] > ema_50_12h_aligned[i]
        trend_down = close[i] < ema_50_12h_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > r1_aligned[i]
        breakout_down = close[i] < s1_aligned[i]
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: breakout above R1, uptrend, volume confirmation
            if breakout_up and trend_up and vol_ok:
                signals[i] = 0.30
                position = 1
            # SHORT: breakout below S1, downtrend, volume confirmation
            elif breakout_down and trend_down and vol_ok:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # EXIT LONG: breakdown below S1 or trend reversal
            if breakout_down or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: breakout above R1 or trend reversal
            if breakout_up or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals