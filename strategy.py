#!/usr/bin/env python3
# 1D_Camarilla_Pivot_R3S3_Breakout_1wTrend_Volume
# Hypothesis: Breakout above Camarilla R3 or below S3 on daily chart with weekly trend filter and volume confirmation.
# Long when: daily close > R3, weekly uptrend (price > weekly EMA50), and volume > 1.5x average.
# Short when: daily close < S3, weekly downtrend (price < weekly EMA50), and volume > 1.5x average.
# Uses volume confirmation to avoid false breakouts and weekly trend to align with higher timeframe momentum.
# Target: 15-25 trades/year per symbol.

name = "1D_Camarilla_Pivot_R3S3_Breakout_1wTrend_Volume"
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
    
    # Daily Camarilla levels (based on previous day)
    prev_close = np.concatenate([[np.nan], close[:-1]])
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    
    # Calculate Camarilla levels
    R3 = prev_close + 1.1 * (prev_high - prev_low) / 1.1
    S3 = prev_close - 1.1 * (prev_high - prev_low) / 1.1
    
    # Volume average (20-period)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = close_1w > ema50_1w
    weekly_downtrend = close_1w < ema50_1w
    
    # Align weekly trend to daily
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or np.isnan(vol_ma[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        weekly_up = weekly_uptrend_aligned[i] > 0.5
        weekly_down = weekly_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: daily close > R3, weekly uptrend, volume confirmation
            if close[i] > R3[i] and weekly_up and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: daily close < S3, weekly downtrend, volume confirmation
            elif close[i] < S3[i] and weekly_down and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below R3 or weekly trend changes
            if close[i] < R3[i] or not weekly_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above S3 or weekly trend changes
            if close[i] > S3[i] or not weekly_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals