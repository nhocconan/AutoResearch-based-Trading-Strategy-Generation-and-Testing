#!/usr/bin/env python3
"""
1d_Donchian_Breakout_WeeklyTrend_VolumeFilter
Hypothesis: Uses weekly Donchian channel breakout on the daily timeframe,
confirmed by weekly EMA50 trend and volume surge. Designed to capture
long-term trends with low trade frequency, working in both bull and bear markets.
"""

name = "1d_Donchian_Breakout_WeeklyTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for Donchian channel and EMA50 (call once before loop)
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 50:
        return np.zeros(n)

    # Calculate weekly Donchian channel (20-period)
    hh_w = df_w['high'].values
    ll_w = df_w['low'].values
    donchian_high = pd.Series(hh_w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(ll_w).rolling(window=20, min_periods=20).min().values

    # Calculate weekly EMA50 for trend filter
    close_w = pd.Series(df_w['close'].values)
    ema50_w = close_w.ewm(span=50, adjust=False, min_periods=50).mean().values

    # Volume confirmation: 5-period average (5 weeks of weekly data)
    vol_avg_5 = pd.Series(df_w['volume'].values).rolling(window=5, min_periods=5).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start from 50 to have enough data for weekly indicators
        # Get aligned values for current daily bar
        donchian_high_aligned = align_htf_to_ltf(prices, df_w, donchian_high)[i]
        donchian_low_aligned = align_htf_to_ltf(prices, df_w, donchian_low)[i]
        ema50_w_aligned = align_htf_to_ltf(prices, df_w, ema50_w)[i]
        vol_avg_val = vol_avg_5[i]
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned) or np.isnan(donchian_low_aligned) or 
            np.isnan(ema50_w_aligned) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above weekly Donchian high with bullish weekly trend and volume surge
            if (close[i] > donchian_high_aligned and 
                close[i] > ema50_w_aligned and 
                volume[i] > vol_avg_val * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below weekly Donchian low with bearish weekly trend and volume surge
            elif (close[i] < donchian_low_aligned and 
                  close[i] < ema50_w_aligned and 
                  volume[i] > vol_avg_val * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below weekly Donchian low or EMA50 (reversal signal)
            if (close[i] < donchian_low_aligned or close[i] < ema50_w_aligned):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above weekly Donchian high or EMA50 (reversal signal)
            if (close[i] > donchian_high_aligned or close[i] > ema50_w_aligned):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals