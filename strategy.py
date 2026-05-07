#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Load daily data for Camarilla pivot
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    w_close = df_1w['close'].values
    ema_50_1w = pd.Series(w_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Camarilla pivot levels from previous day
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    pivot = (d_high + d_low + d_close) / 3
    range_val = d_high - d_low
    r1 = pivot + (range_val * 1.1 / 12)
    s1 = pivot - (range_val * 1.1 / 12)
    
    # Align pivot levels to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Volume spike detection (2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(ema_50_4h[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(pivot_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: break above R1 in weekly uptrend with volume
            if close[i] > r1_4h[i] and ema_50_4h[i] > ema_50_4h[i-1] and vol_condition:
                signals[i] = 0.30
                position = 1
            # Short: break below S1 in weekly downtrend with volume
            elif close[i] < s1_4h[i] and ema_50_4h[i] < ema_50_4h[i-1] and vol_condition:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: price returns to pivot or trend reverses
            if close[i] < pivot_4h[i] or ema_50_4h[i] < ema_50_4h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: price returns to pivot or trend reverses
            if close[i] > pivot_4h[i] or ema_50_4h[i] > ema_50_4h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: Camarilla R1/S1 breakouts with weekly trend filter and volume confirmation
# - Camarilla R1/S1 represent tighter support/resistance levels from previous day
# - Breakout above R1 in weekly uptrend (EMA50 rising) signals bullish continuation
# - Breakdown below S1 in weekly downtrend (EMA50 falling) signals bearish continuation
# - Volume confirmation (2x average) reduces false breakouts
# - Exit when price returns to pivot point or weekly trend reverses
# - Position size 0.30 targets ~30-50 trades/year to avoid fee drag
# - Weekly trend filter provides stronger, more persistent trend than daily
# - Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend)
# - Uses 1w timeframe for trend and 1d for structure, 4h for execution timing
# - R1/S1 levels are more frequently tested than R3/S3, increasing trade frequency while maintaining edge