#!/usr/bin/env python3
name = "4h_Pivot_Reversal_Volume_Trend"
timeframe = "4h"
leverage = 1.0

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
    
    # Daily pivot points (previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivots: P = (H+L+C)/3, S1 = 2P - H, R1 = 2P - L
    prev_high = df_1d['high'].shift(1).values  # Previous day high
    prev_low = df_1d['low'].shift(1).values    # Previous day low
    prev_close = df_1d['close'].shift(1).values # Previous day close
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_high
    s1 = 2 * pivot - prev_low
    
    # Align pivots to 4h timeframe (wait for daily close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily EMA trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h volume spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: price crosses above S1 with volume in uptrend
            if close[i] > s1_aligned[i] and close[i-1] <= s1_aligned[i-1] and vol_condition and ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below R1 with volume in downtrend
            elif close[i] < r1_aligned[i] and close[i-1] >= r1_aligned[i-1] and vol_condition and ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below pivot or trend fails
            if close[i] < pivot_aligned[i] or ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above pivot or trend fails
            if close[i] > pivot_aligned[i] or ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Pivot reversals with daily trend filter and volume confirmation
# - Uses previous day's pivot points (S1, R1, P) as support/resistance levels
# - Long when price breaks above S1 with volume in daily uptrend
# - Short when price breaks below R1 with volume in daily downtrend
# - Daily EMA50 trend filter ensures alignment with higher timeframe trend
# - Volume confirmation (1.5x average) reduces false breakouts
# - Exits when price returns to pivot level or trend fails
# - Position size 0.25 targets ~25-50 trades/year to avoid fee drag
# - Pivot levels provide institutional reference points that work in both bull/bear
# - Works on BTC/ETH as they respect key daily levels during trends and reversals
# - Avoids overtrading by requiring volume confirmation and trend alignment
# - Pivot reversals capture institutional order flow around key daily levels