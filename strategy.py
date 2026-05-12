#!/usr/bin/env python3
# 1h_Camarilla_Pivot_Breakout_4hTrend_Volume
# Hypothesis: Trade 1h breakouts of daily Camarilla pivot levels (R1/S1) aligned with 4h EMA50 trend and volume confirmation.
# Daily Camarilla provides intraday support/resistance levels; 4h EMA filters higher timeframe trend; volume confirms breakout momentum.
# Designed for low frequency (15-37 trades/year) to survive both bull and bear markets by following higher timeframe structure.

name = "1h_Camarilla_Pivot_Breakout_4hTrend_Volume"
timeframe = "1h"
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
    
    # === 4h EMA50 for trend filter ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === Daily Camarilla pivot levels (R1, S1) ===
    # Calculate from daily OHLC (previous completed day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Shift by 1 to use previous day's data
    d_high_prev = np.roll(d_high, 1)
    d_low_prev = np.roll(d_low, 1)
    d_close_prev = np.roll(d_close, 1)
    d_high_prev[0] = np.nan
    d_low_prev[0] = np.nan
    d_close_prev[0] = np.nan
    
    # Camarilla pivot calculation
    pivot = (d_high_prev + d_low_prev + d_close_prev) / 3.0
    range_ = d_high_prev - d_low_prev
    r1 = pivot + (range_ * 1.1 / 12)
    s1 = pivot - (range_ * 1.1 / 12)
    
    # Align daily levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Volume confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 4h EMA50
        trend_up = close[i] > ema_50_4h_aligned[i]
        trend_down = close[i] < ema_50_4h_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > r1_aligned[i]
        breakout_down = close[i] < s1_aligned[i]
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: breakout above R1, uptrend, volume confirmation
            if breakout_up and trend_up and vol_ok:
                signals[i] = 0.20
                position = 1
            # SHORT: breakout below S1, downtrend, volume confirmation
            elif breakout_down and trend_down and vol_ok:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # EXIT LONG: breakdown below S1 or trend reversal
            if breakout_down or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: breakout above R1 or trend reversal
            if breakout_up or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals