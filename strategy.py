#!/usr/bin/env python3
# 4h_Donchian20_Breakout_VolumeTrend_1dEMA50
# Hypothesis: Enter long when price breaks above 4h Donchian upper (20) with volume confirmation and 1d EMA50 uptrend.
# Enter short when price breaks below 4h Donchian lower (20) with volume confirmation and 1d EMA50 downtrend.
# Uses Donchian channel for breakout structure, volume surge for institutional confirmation, and daily EMA50 for trend filter.
# Works in bull (breakouts above upper band in uptrend) and bear (breakdowns below lower band in downtrend).
# Low frequency due to breakout requirement and strict volume + trend confirmation.

name = "4h_Donchian20_Breakout_VolumeTrend_1dEMA50"
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

    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 4h Donchian Channel (20-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * vol_ma_20
    
    # Align daily EMA50 to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required value is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Donchian high + volume confirm + daily uptrend
            if close[i] > donchian_high[i] and volume_confirm[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + volume confirm + daily downtrend
            elif close[i] < donchian_low[i] and volume_confirm[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low OR trend reversal
            if close[i] < donchian_low[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high OR trend reversal
            if close[i] > donchian_high[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals