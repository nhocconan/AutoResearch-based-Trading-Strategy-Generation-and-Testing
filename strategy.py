#!/usr/bin/env python3
# 12h_Donchian_Breakout_20_1dTrend_Volume
# Hypothesis: Breakout above/below 20-period Donchian channel on 12h timeframe in the direction of the daily trend, confirmed by volume spike.
# Donchian channels provide clear breakout levels; daily trend filter ensures alignment with higher timeframe momentum.
# Volume spike confirms institutional participation. Works in bull (breakouts above upper band in uptrend) and bear (breakdowns below lower band in downtrend).
# Low frequency due to reliance on Donchian breakouts and strict volume confirmation.

name = "12h_Donchian_Breakout_20_1dTrend_Volume"
timeframe = "12h"
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

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Daily trend: EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: volume > 2.0 * 24-period average (2 days worth at 12h)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * vol_ma_24
    
    # Donchian channel (20-period) on 12h timeframe
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > upper Donchian + daily uptrend + volume spike
            if close[i] > donchian_high[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < lower Donchian + daily downtrend + volume spike
            elif close[i] < donchian_low[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below lower Donchian OR trend reversal
            if close[i] < donchian_low[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above upper Donchian OR trend reversal
            if close[i] > donchian_high[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals