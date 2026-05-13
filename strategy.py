#!/usr/bin/env python3
# 1d_Donchian_Breakout_TRIX_Trend
# Hypothesis: Enter long when price breaks above Donchian upper channel with TRIX > 0 on weekly timeframe (bullish momentum). 
# Enter short when price breaks below Donchian lower channel with TRIX < 0 on weekly timeframe (bearish momentum).
# Donchian channels provide clear breakout levels. TRIX filters for momentum direction on higher timeframe.
# Works in bull markets (breakouts above upper channel with bullish TRIX) and bear markets (breakdowns below lower channel with bearish TRIX).
# Low frequency due to combined breakout and momentum conditions.

name = "1d_Donchian_Breakout_TRIX_Trend"
timeframe = "1d"
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

    # Get weekly data for TRIX
    df_1w = get_htf_data(prices, '1w')
    
    # Donchian Channel (20-day) on daily timeframe
    donchian_window = 20
    upper_dc = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_dc = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # TRIX on weekly close (15,9,9)
    close_1w = df_1w['close'].values
    ema1 = pd.Series(close_1w).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix_raw = (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix_raw[0] = 0  # First value has no previous
    trix = trix_raw * 100  # Convert to percentage
    
    # Align indicators to daily timeframe
    upper_dc_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), upper_dc)
    lower_dc_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low}), lower_dc)
    trix_aligned = align_htf_to_ltf(prices, df_1w, trix)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(upper_dc_aligned[i]) or 
            np.isnan(lower_dc_aligned[i]) or 
            np.isnan(trix_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above upper Donchian + weekly TRIX > 0
            if close[i] > upper_dc_aligned[i] and trix_aligned[i] > 0:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian + weekly TRIX < 0
            elif close[i] < lower_dc_aligned[i] and trix_aligned[i] < 0:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below lower Donchian channel
            if close[i] < lower_dc_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above upper Donchian channel
            if close[i] > upper_dc_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals