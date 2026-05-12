#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_1wTrend_Volume
Hypothesis: Camarilla pivot levels on 1d combined with 1w trend filter and volume confirmation provides high-probability breakout trades.
Long when price breaks above R1 with 1w uptrend and volume spike, short when price breaks below S1 with 1w downtrend and volume spike.
Designed for 15-25 trades/year on 1d timeframe to work in both bull and bear markets via trend-following breakouts with volume confirmation.
"""

name = "1d_Camarilla_R1S1_Breakout_1wTrend_Volume"
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

    # Get 1w data (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)

    close_1w = df_1w['close'].values

    # Calculate 1w EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)

    # Calculate Camarilla levels for previous day
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    # Shift to get previous day's values
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_typical = np.roll(typical_price, 1)
    # Set first value to NaN (no previous day)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    prev_typical[0] = np.nan

    # Camarilla calculations
    R1 = prev_typical + 1.1 * (prev_high - prev_low) / 12
    S1 = prev_typical - 1.1 * (prev_high - prev_low) / 12

    # Volume confirmation: current volume > 2.0 x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        ema20_val = ema20_1w_aligned[i]
        r1_val = R1[i]
        s1_val = S1[i]
        vol_avg_val = vol_avg_20[i]

        if np.isnan(ema20_val) or np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(vol_avg_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above R1 + 1w uptrend + volume spike
            if close[i] > r1_val and close[i] > ema20_val and volume[i] > vol_avg_val * 2.0:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S1 + 1w downtrend + volume spike
            elif close[i] < s1_val and close[i] < ema20_val and volume[i] > vol_avg_val * 2.0:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below S1 or trend reversal
            if close[i] < s1_val or close[i] < ema20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above R1 or trend reversal
            if close[i] > r1_val or close[i] > ema20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals