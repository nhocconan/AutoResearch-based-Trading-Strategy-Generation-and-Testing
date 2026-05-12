#!/usr/bin/env python3
# 4h Williams Alligator + Volume Spike + Daily EMA Trend
# Hypothesis: Williams Alligator (3 SMAs: Jaw 13, Teeth 8, Lips 5) identifies trend when lines are aligned and separated.
# Jaw below Teeth below Lips = uptrend; Jaw above Teeth above Lips = downtrend.
# Combines with daily EMA50 trend filter and volume spikes for confirmation.
# Works in both bull and bear markets by following Alligator-defined momentum.
# Designed for low trade frequency (~20-40/year) with clear entry/exit rules.

name = "4h_Williams_Alligator_Volume_DailyTrend"
timeframe = "4h"
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
    
    # === Daily Data for EMA Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    daily_close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(daily_close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Williams Alligator (13,8,5) SMAs ===
    # Jaw (13-period SMA of median price)
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    # Teeth (8-period SMA of median price)
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    # Lips (5-period SMA of median price)
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    # === Volume Spike (20-period on 4h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13  # Ensure all indicators ready (max period 13)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_4h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Jaw < Teeth < Lips (uptrend alignment) + volume spike + price above daily EMA50
            if (jaw[i] < teeth[i] and teeth[i] < lips[i] and 
                vol_spike[i] and
                close[i] > ema_50_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Jaw > Teeth > Lips (downtrend alignment) + volume spike + price below daily EMA50
            elif (jaw[i] > teeth[i] and teeth[i] > lips[i] and 
                  vol_spike[i] and
                  close[i] < ema_50_4h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Trend weakens (Alligator lines cross or entangle)
            if not (jaw[i] < teeth[i] and teeth[i] < lips[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend weakens (Alligator lines cross or entangle)
            if not (jaw[i] > teeth[i] and teeth[i] > lips[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals