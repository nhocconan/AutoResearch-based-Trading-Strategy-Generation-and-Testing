#!/usr/bin/env python3
name = "6h_Weekly_Pivot_D1_Trend_Confirmation"
timeframe = "6h"
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
    
    # Load weekly data ONCE for pivot points and trend filter
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 2:
        return np.zeros(n)
    
    # Weekly pivot points from previous week
    w_high = df_w['high'].values
    w_low = df_w['low'].values
    w_close = df_w['close'].values
    
    pivot_w = (w_high + w_low + w_close) / 3
    range_w = w_high - w_low
    r1 = pivot_w + (range_w * 1.0 / 3)
    s1 = pivot_w - (range_w * 1.0 / 3)
    r2 = pivot_w + (range_w * 2.0 / 3)
    s2 = pivot_w - (range_w * 2.0 / 3)
    
    # Align weekly pivot levels to 6h timeframe
    r1_6h = align_htf_to_ltf(prices, df_w, r1)
    s1_6h = align_htf_to_ltf(prices, df_w, s1)
    r2_6h = align_htf_to_ltf(prices, df_w, r2)
    s2_6h = align_htf_to_ltf(prices, df_w, s2)
    pivot_6h = align_htf_to_ltf(prices, df_w, pivot_w)
    
    # Weekly EMA13 for trend filter
    ema_13_w = pd.Series(w_close).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_6h = align_htf_to_ltf(prices, df_w, ema_13_w)
    
    # Volume spike detection (1.8x 24-period average)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 13)
    
    for i in range(start_idx, n):
        if (np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or np.isnan(r2_6h[i]) or 
            np.isnan(s2_6h[i]) or np.isnan(pivot_6h[i]) or np.isnan(ema_13_6h[i]) or 
            np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_24[i] * 1.8
        
        if position == 0:
            # Long: break above R1 in weekly uptrend with volume
            if close[i] > r1_6h[i] and ema_13_6h[i] > ema_13_6h[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 in weekly downtrend with volume
            elif close[i] < s1_6h[i] and ema_13_6h[i] < ema_13_6h[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to weekly pivot or trend reverses
            if close[i] < pivot_6h[i] or ema_13_6h[i] < ema_13_6h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to weekly pivot or trend reverses
            if close[i] > pivot_6h[i] or ema_13_6h[i] > ema_13_6h[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot breakouts with weekly trend filter and volume confirmation
# - Weekly R1/S1 act as significant support/resistance from prior week
# - Breakout above R1 in weekly uptrend (EMA13 rising) signals bullish continuation
# - Breakdown below S1 in weekly downtrend (EMA13 falling) signals bearish continuation
# - Volume confirmation (1.8x average) reduces false breakouts
# - Exit when price returns to weekly pivot or weekly trend reverses
# - Position size 0.25 targets ~20-40 trades/year to avoid fee drag
# - Weekly timeframe provides structural context for 6h entries
# - Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend)
# - Focus on BTC/ETH as primary targets (weekly structure meaningful for major pairs)