#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume confirmation.
Long when price breaks above 4h Camarilla R1 level AND 12h close > 12h EMA50 (uptrend) AND volume > 2.0x 20-period MA.
Short when price breaks below 4h Camarilla S1 level AND 12h close < 12h EMA50 (downtrend) AND volume > 2.0x 20-period MA.
Exit when price retraces to the 4h Camarilla pivot point or 12h trend reverses.
Designed for low trade frequency (target: 25-40/year) with strong structure from proven Camarilla patterns.
Camarilla levels provide tighter, more meaningful breakouts than Donchian in ranging/volatile markets.
Volume filter set high (2.0x) to reduce false breakouts and overtrading.
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
    
    # Calculate 4h Camarilla levels (R1, S1, pivot) from previous bar
    # Camarilla: based on previous bar's range
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan  # first bar has no previous
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate pivot and levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    camarilla_r1 = pivot + range_hl * 1.1 / 12.0  # R1 = pivot + (high-low)*1.1/12
    camarilla_s1 = pivot - range_hl * 1.1 / 12.0  # S1 = pivot - (high-low)*1.1/12
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 4h volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready (need at least 50 for EMA, plus lookbacks)
    start_idx = max(50, 20) + 1  # +1 for the roll(1) shift
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(pivot[i]) or np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 12h close > EMA50 = uptrend, close < EMA50 = downtrend
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        trend_up = close_12h_aligned[i] > ema_50_12h_aligned[i]
        trend_down = close_12h_aligned[i] < ema_50_12h_aligned[i]
        
        # Volume filter: 4h volume > 2.0x 20-period MA (stricter to reduce trades)
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R1 AND uptrend AND volume filter
            if close[i] > camarilla_r1[i] and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1 AND downtrend AND volume filter
            elif close[i] < camarilla_s1[i] and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price retraces to Camarilla pivot OR 12h trend turns down
                if close[i] <= pivot[i] or not trend_up:
                    exit_signal = True
            elif position == -1:
                # Short exit: price retraces to Camarilla pivot OR 12h trend turns up
                if close[i] >= pivot[i] or not trend_down:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R1S1_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0