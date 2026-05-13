#!/usr/bin/env python3
# 4h_Donchian_20_Breakout_Trend_Volume_Spike
# Hypothesis: Enter long when price breaks above 4h Donchian(20) high, with 1d EMA50 uptrend and volume spike; short when price breaks below Donchian(20) low, with 1d EMA50 downtrend and volume spike. Exit on opposite Donchian break or trend reversal. Trend filter ensures alignment with higher timeframe momentum, reducing false breakouts. Volume surge confirms institutional participation. Works in bull (breakouts above in uptrend) and bear (breakdowns below in downtrend). Designed for low frequency (<50 trades/year) to minimize fee drag.

name = "4h_Donchian_20_Breakout_Trend_Volume_Spike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for trend
    df_1d = get_htf_data(prices, '1d')
    
    # Daily trend: EMA50
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 4h Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align daily indicators to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: volume > 2.0 * 6-period average (1 day worth at 4h)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_spike = volume > 2.0 * vol_ma_6
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above Donchian high + daily uptrend + volume spike
            if close[i] > high_20[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Donchian low + daily downtrend + volume spike
            elif close[i] < low_20[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Break below Donchian low OR trend reversal
            if close[i] < low_20[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Break above Donchian high OR trend reversal
            if close[i] > high_20[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals