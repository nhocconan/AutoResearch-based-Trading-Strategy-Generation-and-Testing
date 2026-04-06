#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Bull/Bear Power with 1-day Trend Filter.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low.
# Enter long when Bull Power > 0 and 1-day EMA50 > EMA200 (uptrend).
# Enter short when Bear Power > 0 and 1-day EMA50 < EMA200 (downtrend).
# Exit when power reverses sign or trend changes.
# Works in both bull and bear markets via trend-aligned power signals.
# Target: 50-150 trades over 4 years (12-37/year).

name = "6h_elder_ray_trend_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate EMA13 for Elder Ray (using close)
    ema13 = np.full(n, np.nan)
    ema13_prev = np.nan
    alpha = 2.0 / (13 + 1)
    for i in range(n):
        if np.isnan(ema13_prev):
            ema13[i] = close[i]
        else:
            ema13[i] = alpha * close[i] + (1 - alpha) * ema13_prev
        ema13_prev = ema13[i]
    
    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Daily EMA50 and EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # EMA50 on daily
    ema50_1d = np.full(len(close_1d), np.nan)
    ema50_prev = np.nan
    alpha50 = 2.0 / (50 + 1)
    for i in range(len(close_1d)):
        if np.isnan(ema50_prev):
            ema50_1d[i] = close_1d[i]
        else:
            ema50_1d[i] = alpha50 * close_1d[i] + (1 - alpha50) * ema50_prev
        ema50_prev = ema50_1d[i]
    
    # EMA200 on daily
    ema200_1d = np.full(len(close_1d), np.nan)
    ema200_prev = np.nan
    alpha200 = 2.0 / (200 + 1)
    for i in range(len(close_1d)):
        if np.isnan(ema200_prev):
            ema200_1d[i] = close_1d[i]
        else:
            ema200_1d[i] = alpha200 * close_1d[i] + (1 - alpha200) * ema200_prev
        ema200_prev = ema200_1d[i]
    
    # Trend: 1 if EMA50 > EMA200, -1 if EMA50 < EMA200
    trend_1d = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        if not (np.isnan(ema50_1d[i]) or np.isnan(ema200_1d[i])):
            if ema50_1d[i] > ema200_1d[i]:
                trend_1d[i] = 1
            elif ema50_1d[i] < ema200_1d[i]:
                trend_1d[i] = -1
            else:
                trend_1d[i] = 0
    
    # Align trend to 6h timeframe (shifted by 1 daily bar)
    trend_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if trend data not available
        if np.isnan(trend_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: Bull Power <= 0 or trend turns down
            if (bull_power[i] <= 0 or trend_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Bear Power <= 0 or trend turns up
            if (bear_power[i] <= 0 or trend_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend alignment
            if trend_aligned[i] == 1:  # uptrend
                # Enter long when Bull Power > 0
                if bull_power[i] > 0:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
            elif trend_aligned[i] == -1:  # downtrend
                # Enter short when Bear Power > 0
                if bear_power[i] > 0:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals