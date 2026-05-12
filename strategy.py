#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
Hypothesis: For 1h timeframe, use 4h Camarilla levels (R1/S1) for structure and 4h EMA50 for trend.
Enter long when price breaks above R1 with 4h uptrend and volume confirmation; short when breaks below S1 with 4h downtrend.
Use 1h only for entry timing precision. Target 15-37 trades/year to avoid fee drag.
Works in bull (breakouts) and bear (mean reversion at S1/R1 in ranging markets).
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
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

    # Get 4h data (call once before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)

    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values

    # Calculate Camarilla pivot levels for previous 4h bar
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    hl_range_4h = high_4h - low_4h
    r1_4h = close_4h + hl_range_4h * 1.1 / 12
    s1_4h = close_4h - hl_range_4h * 1.1 / 12

    # Align to 1h timeframe (values from previous 4h bar's close)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)

    # Calculate 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)

    # Volume confirmation: 1h volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        r1_val = r1_4h_aligned[i]
        s1_val = s1_4h_aligned[i]
        ema50_val = ema50_4h_aligned[i]
        vol_avg_val = vol_avg_20[i]

        if np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(ema50_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above R1 + 4h uptrend + volume confirmation
            if close[i] > r1_val and close[i] > ema50_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = 0.20
                position = 1
            # SHORT: Close below S1 + 4h downtrend + volume confirmation
            elif close[i] < s1_val and close[i] < ema50_val and volume[i] > vol_avg_val * 1.5:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below EMA50 or 4h S1 (mean reversion)
            if close[i] < ema50_val or close[i] < s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Close above EMA50 or 4h R1 (mean reversion)
            if close[i] > ema50_val or close[i] > r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals