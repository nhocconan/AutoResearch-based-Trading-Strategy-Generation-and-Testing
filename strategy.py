#!/usr/bin/env python3
"""
12h_1D_Camarilla_R1_S1_Breakout_Trend_Volume_Filter
Hypothesis: On 12h timeframe, price breaking above Camarilla R1 or below S1 levels
from the prior 1d candle, with 1d volume > 1.5x 20-period average and price above/below
1d EMA34 for trend filter, captures sustained moves in both bull and bear markets.
Designed for 12-37 trades/year to minimize fee drag while capitalizing on institutional
levels and volume confirmation. Uses 1d timeframe for Camarilla levels, EMA trend,
and volume average to avoid look-ahead and ensure robustness.
"""

name = "12h_1D_Camarilla_R1_S1_Breakout_Trend_Volume_Filter"
timeframe = "12h"
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

    # Get 1d data (call once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)

    # Calculate 1d Camarilla levels: R1, S1
    # Based on prior day's high, low, close
    ph = df_1d['high'].values
    pl = df_1d['low'].values
    pc = df_1d['close'].values

    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    r1 = pc + (ph - pl) * 1.1 / 12
    s1 = pc - (ph - pl) * 1.1 / 12

    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values

    # Calculate 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values

    # Align all 1d indicators to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # warmup for EMA34 and vol avg
        # Skip if any aligned value is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Current 1d volume (aligned)
        vol_1d_current = volume_1d[i // 288]  # 288 12h bars per 1d (24h*60/12min)
        vol_avg_val = vol_avg_20_1d_aligned[i]

        if position == 0:
            # LONG: Price above R1, volume surge, and price above EMA34 (uptrend)
            if (close[i] > r1_aligned[i] and
                vol_1d_current > vol_avg_val * 1.5 and
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below S1, volume surge, and price below EMA34 (downtrend)
            elif (close[i] < s1_aligned[i] and
                  vol_1d_current > vol_avg_val * 1.5 and
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below S1 (reversal) or below EMA34 (trend change)
            if close[i] < s1_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above R1 (reversal) or above EMA34 (trend change)
            if close[i] > r1_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals