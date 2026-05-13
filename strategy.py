#!/usr/bin/env python3
# 12h_Donchian_Breakout_WeeklyTrend_DailyVolume
# Hypothesis: Enter long when price breaks above 20-period Donchian high with weekly uptrend and daily volume spike.
# Enter short when price breaks below 20-period Donchian low with weekly downtrend and daily volume spike.
# Weekly trend filter reduces false breakouts in counter-trend moves. Daily volume confirms institutional participation.
# Works in bull (breakouts above upper band in uptrend) and bear (breakdowns below lower band in downtrend).
# Low frequency due to Donchian breakout requirement and multi-timeframe alignment.

name = "12h_Donchian_Breakout_WeeklyTrend_DailyVolume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for trend
    df_1w = get_htf_data(prices, '1w')
    
    # Get daily data for volume
    df_1d = get_htf_data(prices, '1d')
    
    # Donchian channels (20-period) on 12h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly trend: EMA50
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily volume spike: volume > 2.0 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    # Align weekly and daily indicators to 12h timeframe
    highest_20_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low, 'close': close}), highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low, 'close': close}), lowest_20)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(highest_20_aligned[i]) or 
            np.isnan(lowest_20_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above Donchian high + weekly uptrend + volume spike
            if close[i] > highest_20_aligned[i] and close[i] > ema50_1w_aligned[i] and volume_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Donchian low + weekly downtrend + volume spike
            elif close[i] < lowest_20_aligned[i] and close[i] < ema50_1w_aligned[i] and volume_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below Donchian low OR weekly trend reversal
            if close[i] < lowest_20_aligned[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above Donchian high OR weekly trend reversal
            if close[i] > highest_20_aligned[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals