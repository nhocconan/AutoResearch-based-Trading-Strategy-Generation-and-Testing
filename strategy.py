#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume
Hypothesis: Camarilla pivot levels on 1h provide precise entry/exit points, filtered by 4h trend direction and 1d volume confirmation. This combination works in both bull and bear markets by only taking trades in the direction of the 4h trend, with volume ensuring conviction. The strategy targets 15-30 trades per year to avoid fee drag.
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
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

    # Get 4h data for trend filter (EMA34)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    close_4h = df_4h['close'].values

    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values

    # Calculate 4h EMA34 for trend filter
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)

    # Calculate 1d average volume (20-period) for volume filter
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)

    # Calculate Camarilla levels for 1h (using previous bar's high/low/close)
    # Camarilla: H4 = Close + 1.5*(High-Low), L4 = Close - 1.5*(High-Low)
    # We'll use R1/S1 levels: R1 = Close + 1.1*(High-Low)/12, S1 = Close - 1.1*(High-Low)/12
    # But we need previous bar's values, so we shift
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan

    rng = prev_high - prev_low
    r1 = prev_close + 1.1 * rng / 12
    s1 = prev_close - 1.1 * rng / 12

    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(1, n):  # Start from 1 to have previous bar data
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(ema34_4h_aligned[i]) or 
            np.isnan(vol_avg_20_1d_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + 4h uptrend + volume confirmation
            if (close[i] > r1[i] and 
                close[i] > ema34_4h_aligned[i] and  # Price above 4h EMA34 = uptrend
                volume[i] > vol_avg_20_1d_aligned[i] * 1.5):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 + 4h downtrend + volume confirmation
            elif (close[i] < s1[i] and 
                  close[i] < ema34_4h_aligned[i] and  # Price below 4h EMA34 = downtrend
                  volume[i] > vol_avg_20_1d_aligned[i] * 1.5):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or 4h trend turns down
            if (close[i] < s1[i] or 
                close[i] < ema34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or 4h trend turns up
            if (close[i] > r1[i] or 
                close[i] > ema34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals