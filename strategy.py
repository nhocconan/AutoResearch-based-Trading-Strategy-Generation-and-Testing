#!/usr/bin/env python3
# 12h_Donchian20_TrendPlus
# Hypothesis: Use 20-bar Donchian channels on 12h timeframe with 1-day trend filter and volume confirmation.
# Long when price breaks above upper Donchian channel, volume spikes, and 1-day EMA50 is rising.
# Short when price breaks below lower Donchian channel, volume spikes, and 1-day EMA50 is falling.
# Exit when price crosses the opposite Donchian band or trend reverses.
# Designed for low turnover (~15-30/year) to avoid fee drag while capturing major trends.

name = "12h_Donchian20_TrendPlus"
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

    # Get 12h data for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper band: highest high of last 20 periods
    donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to lower timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Volume confirmation: current volume > 2.0 x 30-period average
    vol_ma = np.full(n, np.nan)
    for i in range(30, n):
        vol_ma[i] = np.mean(volume[i-30:i])
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):
        # Skip if data is not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above upper Donchian with volume spike and uptrend
            if close[i] > donchian_upper_aligned[i] and volume_spike[i] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below lower Donchian with volume spike and downtrend
            elif close[i] < donchian_lower_aligned[i] and volume_spike[i] and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below lower Donchian or trend turns down
            if close[i] < donchian_lower_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above upper Donchian or trend turns up
            if close[i] > donchian_upper_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals