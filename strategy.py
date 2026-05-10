#!/usr/bin/env python3
name = "1d_Camarilla_R1_S1_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous day
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Camarilla R1 and S1 levels
    camarilla_r1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    camarilla_s1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Weekly trend filter (1w close > 20-period EMA)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema20_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema20_1w[19] = np.mean(close_1w[:20])
        alpha = 2 / (20 + 1)
        for i in range(20, len(close_1w)):
            ema20_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema20_1w[i-1]
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume confirmation: current volume > 1.5x 20-day average volume
    vol_sma20 = np.full(n, np.nan)
    if n >= 20:
        vol_sma20[19] = np.mean(volume[:20])
        for i in range(20, n):
            vol_sma20[i] = (vol_sma20[i-1] * 19 + volume[i]) / 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need previous day data
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or \
           np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_sma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * vol_sma20[i]
        
        # Trend filter: price above/below weekly EMA20
        is_uptrend = close[i] > ema20_1w_aligned[i]
        is_downtrend = close[i] < ema20_1w_aligned[i]
        
        if position == 0:
            # Long when price breaks above Camarilla R1 in uptrend with volume
            if close[i] > camarilla_r1[i] and is_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Camarilla S1 in downtrend with volume
            elif close[i] < camarilla_s1[i] and is_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Camarilla S1 or trend changes
            if close[i] < camarilla_s1[i] or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Camarilla R1 or trend changes
            if close[i] > camarilla_r1[i] or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals