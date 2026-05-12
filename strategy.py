#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_Volume_Trend
Hypothesis: On 12h timeframe, Donchian channel breakouts (20-period) from prior period highs/lows capture momentum in both bull and bear markets.
Volume confirmation (>1.5x 20-period average) ensures breakouts are genuine.
Trend filter uses 1d EMA34 to align with higher timeframe trend: only long when price > EMA34, short when price < EMA34.
This combination reduces false breakouts and adapts to market regimes while maintaining low trade frequency.
"""

name = "12h_Donchian20_Breakout_Volume_Trend"
timeframe = "12h"
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

    # Get 1d data (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Calculate Donchian channels (20-period) on 12h data
    # Upper band = highest high of last 20 periods, Lower band = lowest low of last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_band = high_series.rolling(window=20, min_periods=20).max().values
    lower_band = low_series.rolling(window=20, min_periods=20).min().values

    # Volume confirmation: 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_avg_20 = volume_series.rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Get aligned values for current 12h bar
        ema34 = ema34_1d_aligned[i]
        upper_band_val = upper_band[i]
        lower_band_val = lower_band[i]
        vol_avg_val = vol_avg_20[i]

        # Skip if any required data is NaN
        if (np.isnan(ema34) or np.isnan(upper_band_val) or 
            np.isnan(lower_band_val) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above upper band + price above EMA34 + volume surge
            if (close[i] > upper_band_val and 
                close[i] > ema34 and 
                volume[i] > vol_avg_val * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower band + price below EMA34 + volume surge
            elif (close[i] < lower_band_val and 
                  close[i] < ema34 and 
                  volume[i] > vol_avg_val * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below lower band or price below EMA34
            if (close[i] < lower_band_val or close[i] < ema34):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above upper band or price above EMA34
            if (close[i] > upper_band_val or close[i] > ema34):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals