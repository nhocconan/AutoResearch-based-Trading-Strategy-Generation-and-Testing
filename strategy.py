#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_Pivot_R1S1_Breakout_Volume_Trend_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data once before loop
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d pivot levels from previous 1d bar
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = np.nan
    prev_high_1d = np.roll(high_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d = np.roll(low_1d, 1)
    prev_low_1d[0] = np.nan
    
    # Pivot = (H + L + C) / 3
    pivot_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    # R1 = C + (H - L) * 1.1 / 12
    r1_1d = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 12.0
    # S1 = C - (H - L) * 1.1 / 12
    s1_1d = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 12.0
    
    # Calculate 1w trend: close above/below 21-period EMA
    close_1w_series = pd.Series(close_1w)
    ema21_1w = close_1w_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align to 4h timeframe
    pivot_1d_4h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_4h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_4h = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema21_1w_4h = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Volume confirmation: current volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30
    
    for i in range(start_idx, n):
        if np.isnan(pivot_1d_4h[i]) or np.isnan(r1_1d_4h[i]) or np.isnan(s1_1d_4h[i]) or \
           np.isnan(ema21_1w_4h[i]) or np.isnan(vol_ma_30[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_30[i]
        
        # Volume spike: current volume > 1.8x average
        volume_spike = vol > 1.8 * vol_ma
        
        # Trend filter: only trade in direction of 1w EMA21
        trend_up = price > ema21_1w_4h[i]
        trend_down = price < ema21_1w_4h[i]
        
        if position == 0:
            # Long: Price breaks above 1d R1 with volume spike and uptrend
            if price > r1_1d_4h[i] and volume_spike and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 1d S1 with volume spike and downtrend
            elif price < s1_1d_4h[i] and volume_spike and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price returns below 1d S1 (reversal signal)
            if price < s1_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price returns above 1d R1 (reversal signal)
            if price > r1_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals