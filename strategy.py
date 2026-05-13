#!/usr/bin/env python3
# 12h_Donchian_Breakout_Volume_Trend
# Hypothesis: Breakout above/below 20-period Donchian channel on 12h chart with volume confirmation and daily trend filter. Works in bull (breakouts in uptrend) and bear (breakdowns in downtrend) by capturing institutional moves. Volume confirms breakout legitimacy, trend filter avoids false signals in chop. Low frequency due to Donchian breakout rarity + volume confirmation.

name = "12h_Donchian_Breakout_Volume_Trend"
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

    # Get daily data for Donchian channels and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Donchian Channel (20-period high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    upper_dc = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_dc = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily trend: EMA50
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 12h timeframe
    upper_dc_aligned = align_htf_to_ltf(prices, df_1d, upper_dc)
    lower_dc_aligned = align_htf_to_ltf(prices, df_1d, lower_dc)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: volume > 1.5 * 2-period average (1 day worth at 12h)
    vol_ma_2 = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    volume_spike = volume > 1.5 * vol_ma_2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(upper_dc_aligned[i]) or 
            np.isnan(lower_dc_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > upper Donchian + daily uptrend + volume spike
            if close[i] > upper_dc_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.30
                position = 1
            # SHORT: Close < lower Donchian + daily downtrend + volume spike
            elif close[i] < lower_dc_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below lower Donchian OR trend reversal
            if close[i] < lower_dc_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Close above upper Donchian OR trend reversal
            if close[i] > upper_dc_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30

    return signals