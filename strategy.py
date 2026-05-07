#!/usr/bin/env python3
name = "6h_Weekly_Pivot_Swing_1dTrend_Volume"
timeframe = "6h"
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
    
    # Load daily data ONCE for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load weekly data ONCE for weekly pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Daily OHLC for pivot calculation
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Weekly OHLC for weekly pivot
    w_high = df_1w['high'].values
    w_low = df_1w['low'].values
    w_close = df_1w['close'].values
    
    # Daily pivot levels (standard)
    d_pivot = (d_high + d_low + d_close) / 3
    d_range = d_high - d_low
    d_r1 = d_pivot + (d_range * 1.0)
    d_s1 = d_pivot - (d_range * 1.0)
    d_r2 = d_pivot + (d_range * 2.0)
    d_s2 = d_pivot - (d_range * 2.0)
    
    # Weekly pivot levels
    w_pivot = (w_high + w_low + w_close) / 3
    w_range = w_high - w_low
    w_r1 = w_pivot + (w_range * 1.0)
    w_s1 = w_pivot - (w_range * 1.0)
    
    # Align daily levels to 6h
    d_r1_6h = align_htf_to_ltf(prices, df_1d, d_r1)
    d_s1_6h = align_htf_to_ltf(prices, df_1d, d_s1)
    d_r2_6h = align_htf_to_ltf(prices, df_1d, d_r2)
    d_s2_6h = align_htf_to_ltf(prices, df_1d, d_s2)
    d_pivot_6h = align_htf_to_ltf(prices, df_1d, d_pivot)
    
    # Align weekly levels to 6h
    w_r1_6h = align_htf_to_ltf(prices, df_1w, w_r1)
    w_s1_6h = align_htf_to_ltf(prices, df_1w, w_s1)
    w_pivot_6h = align_htf_to_ltf(prices, df_1w, w_pivot)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (1.5x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(d_r1_6h[i]) or np.isnan(d_s1_6h[i]) or 
            np.isnan(w_r1_6h[i]) or np.isnan(w_s1_6h[i]) or
            np.isnan(ema_34_6h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: bounce from daily S1 with weekly bullish bias and volume
            if (close[i] > d_s1_6h[i] and close[i] <= d_s1_6h[i] * 1.02 and  # near S1
                ema_34_6h[i] > ema_34_6h[i-1] and  # daily uptrend
                close[i] > w_pivot_6h[i] and  # above weekly pivot
                vol_condition):
                signals[i] = 0.25
                position = 1
            # Short: rejection at daily R1 with weekly bearish bias and volume
            elif (close[i] < d_r1_6h[i] and close[i] >= d_r1_6h[i] * 0.98 and  # near R1
                  ema_34_6h[i] < ema_34_6h[i-1] and  # daily downtrend
                  close[i] < w_pivot_6h[i] and  # below weekly pivot
                  vol_condition):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price reaches daily R1 or weekly R1 or trend reverses
            if (close[i] >= d_r1_6h[i] or 
                close[i] >= w_r1_6h[i] or
                ema_34_6h[i] < ema_34_6h[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price reaches daily S1 or weekly S1 or trend reverses
            if (close[i] <= d_s1_6h[i] or 
                close[i] <= w_s1_6h[i] or
                ema_34_6h[i] > ema_34_6h[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot swing strategy for 6B timeframe
# - Uses daily S1/R1 as swing levels for mean reversion entries
# - Weekly pivot acts as bias filter: only long above weekly pivot, short below
# - Daily EMA34 trend filter ensures we trade with the higher timeframe trend
# - Volume confirmation (1.5x average) reduces false signals
# - Entries occur near daily support/resistance with trend alignment
# - Exits at opposing daily/weekly pivot levels or trend reversal
# - Position size 0.25 balances risk and return
# - Designed to work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend)
# - Weekly context prevents trading against the larger trend
# - Target: 50-120 trades over 4 years (12-30/year) to minimize fee drag