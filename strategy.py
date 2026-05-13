#!/usr/bin/env python3
# 1d_Donchian_Breakout_WeeklyTrend_Volume
# Hypothesis: Enter long when price breaks above weekly Donchian high (20-week high) on daily close,
# with weekly trend aligned (price above weekly EMA40) and volume confirmation.
# Enter short when price breaks below weekly Donchian low (20-week low) with weekly downtrend
# and volume confirmation. Uses weekly timeframe for trend and structure, daily for execution.
# Trend filter reduces false breakouts in choppy markets. Volume surge confirms institutional
# participation. Designed to work in both bull (breakouts in uptrend) and bear (breakdowns in downtrend).
# Low frequency due to weekly Donchian breakout requirement and strict volume confirmation.

name = "1d_Donchian_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
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

    # Get weekly data for Donchian channels and trend
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly Donchian Channel (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Weekly trend: EMA40
    close_1w = df_1w['close'].values
    ema40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    
    # Align weekly indicators to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    # Volume spike: volume > 2.0 * 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema40_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > weekly Donchian high + weekly uptrend + volume spike
            if close[i] > donchian_high_aligned[i] and close[i] > ema40_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < weekly Donchian low + weekly downtrend + volume spike
            elif close[i] < donchian_low_aligned[i] and close[i] < ema40_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below weekly EMA40 (trend reversal)
            if close[i] < ema40_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above weekly EMA40 (trend reversal)
            if close[i] > ema40_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals