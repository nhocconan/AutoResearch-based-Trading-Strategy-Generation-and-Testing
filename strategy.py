#!/usr/bin/env python3
# 12h_Donchian_Breakout_20_Trend_Volume
# Hypothesis: Price breaks out of 20-period Donchian channel on 12h timeframe with trend confirmation and volume spike.
# Long when price breaks above upper Donchian band with 12h uptrend and volume spike.
# Short when price breaks below lower Donchian band with 12h downtrend and volume spike.
# Trend filter uses 12h EMA50 to ensure alignment with higher timeframe momentum.
# Volume confirmation requires volume > 2.0 * 3-period average to reduce false breakouts.
# Works in bull markets (breakouts above upper band in uptrend) and bear markets (breakdowns below lower band in downtrend).
# Target: 15-30 trades/year per symbol to minimize fee drag.

name = "12h_Donchian_Breakout_20_Trend_Volume"
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

    # Get 1d data for Donchian channel calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channel (20-period) from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper band: highest high of last 20 days
    upper_band = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 days
    lower_band = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # 1d trend: EMA50
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 12h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: volume > 2.0 * 3-period average (1.5 days worth at 12h)
    vol_ma_3 = pd.Series(volume).rolling(window=3, min_periods=3).mean().values
    volume_spike = volume > 2.0 * vol_ma_3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > upper band + 1d uptrend + volume spike
            if close[i] > upper_band_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < lower band + 1d downtrend + volume spike
            elif close[i] < lower_band_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below lower band or trend reversal
            if close[i] < lower_band_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above upper band or trend reversal
            if close[i] > upper_band_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals