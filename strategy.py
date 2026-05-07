#!/usr/bin/env python3
name = "12h_KAMA_1dTrend_Volume_Confirm"
timeframe = "12h"
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
    
    # Load daily data ONCE before loop for trend filter and KAMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on daily timeframe
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    abs_change = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        if close_1d[i] != close_1d[i-1]:
            er[i] = np.abs(close_1d[i] - close_1d[i-10]) / np.sum(abs_change[i-9:i+1]) if i >= 10 else 0
        else:
            er[i] = 0
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # ER smoothing with fast=2, slow=30
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 2-period average (1 day of 12h bars)
    vol_ma_2 = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 2)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(kama_1d_aligned[i]) or 
            np.isnan(vol_ma_2[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_2[i] * 2.0
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > kama_1d_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA with volume and daily downtrend
            elif close[i] < kama_1d_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below KAMA or volume drops
            if close[i] < kama_1d_aligned[i] or volume[i] < vol_ma_2[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above KAMA or volume drops
            if close[i] > kama_1d_aligned[i] or volume[i] < vol_ma_2[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h KAMA with 1d trend and volume confirmation
# - KAUFMAN ADAPTIVE MOVING AVERAGE adapts to market noise, reducing false signals
# - Long when price crosses above KAMA with volume spike in daily uptrend
# - Short when price crosses below KAMA with volume spike in daily downtrend
# - Volume spike (2.0x average) confirms institutional participation
# - Works in both bull (buy KAMA breaks in uptrend) and bear (sell KAMA breaks in downtrend)
# - Exit when price returns to KAMA or volume weakens
# - Position size 0.25 targets ~20-50 trades/year, avoiding fee drag
# - Uses daily KAMA for noise reduction vs simple MA
# - Daily trend filter reduces whipsaws vs using same timeframe
# - Designed to work in BOTH bull and bear markets via trend filter
# - Volume confirmation reduces false breakouts
# - Novel combination: KAMA (1d) + trend (1d) + volume (12h) not recently tried
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits