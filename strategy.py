#!/usr/bin/env python3
# 1d_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Buy breakouts above weekly R1 in long-term uptrends (weekly close > EMA50) and sell breakdowns below S1 in downtrends (weekly close < EMA50), with volume confirmation. Uses 1w for trend and structure, 1d for entry timing. Targets 30-100 trades over 4 years via strict multi-condition entry. Works in bull/bear by requiring trend alignment, avoiding counter-trend traps.

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend and structure
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly Camarilla levels from previous week
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    prev_close = df_1w['close'].shift(1).values
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align weekly Camarilla levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume average (20-period for daily = ~1 month)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough history for weekly Camarilla (previous week) + EMA + vol MA
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend: weekly close > EMA50
        close_1w_aligned = align_htf_to_ltf(prices, df_1w, df_1w['close'].values)
        uptrend = close_1w_aligned[i] > ema_50_1w_aligned[i]
        downtrend = close_1w_aligned[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation (2x average)
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Breakout above R1 in uptrend with volume
            if close[i] > r1_aligned[i] and uptrend and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below S1 in downtrend with volume
            elif close[i] < s1_aligned[i] and downtrend and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Long exit: close below R1 or trend fails
                if close[i] < r1_aligned[i] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: close above S1 or trend fails
                if close[i] > s1_aligned[i] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals