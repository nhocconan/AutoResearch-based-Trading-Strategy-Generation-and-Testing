#!/usr/bin/env python3
# 4h_Pivot_Breakout_1dEMA34_Trend_Volume
# Hypothesis: Use daily pivot point (PP) with breakouts above R1 or below S1, filtered by 1d EMA34 trend and volume spike.
# Long when price breaks above daily R1 in uptrend (price > EMA34) with volume > 1.5x 20-bar average.
# Short when price breaks below daily S1 in downtrend (price < EMA34) with volume spike.
# Exit when price returns to daily pivot point or trend reverses.
# Daily pivots provide institutional support/resistance; EMA34 filters trend; volume confirms breakout.
# Designed for 20-40 trades/year to avoid fee drag, targeting 80-160 total trades over 4 years.

name = "4h_Pivot_Breakout_1dEMA34_Trend_Volume"
timeframe = "4h"
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

    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot points: R1, S1, PP
    # Based on previous day's OHLC
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    
    pp_1d = (high_1d + low_1d + close_1d) / 3
    r1_1d = 2 * pp_1d - low_1d
    s1_1d = 2 * pp_1d - high_1d
    
    # Align daily pivot levels to 4h timeframe
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d.values)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d.values)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d.values)

    # Get daily data for EMA trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):
        # Skip if any required value is NaN
        if (np.isnan(pp_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + price above 1d EMA34 (uptrend) + volume spike
            if (close[i] > r1_1d_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + price below 1d EMA34 (downtrend) + volume spike
            elif (close[i] < s1_1d_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to pivot point (PP) or trend changes (price below EMA34)
            if (close[i] <= pp_1d_aligned[i] or close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to pivot point (PP) or trend changes (price above EMA34)
            if (close[i] >= pp_1d_aligned[i] or close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals