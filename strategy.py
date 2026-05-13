#/usr/bin/env python3
# 1d_1w_Camarilla_R1S1_Breakout_Trend_Volume
# Hypothesis: On daily timeframe, breakout beyond weekly Camarilla R1/S1 levels (weekly support/resistance)
# with alignment to 1d trend (price vs EMA20) and volume confirmation captures strong momentum moves.
# Weekly trend provides long-term filter reducing whipsaws in chop. Volume spike confirms institutional participation.
# Designed for low-frequency, high-quality setups with minimal trade frequency to avoid fee drag.

name = "1d_1w_Camarilla_R1S1_Breakout_Trend_Volume"
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

    # Get weekly data for trend filter and Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values

    # Calculate weekly EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)

    # Calculate weekly Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = close_1w + (high_1w - low_1w) * 1.1 / 12.0
    s1_1w = close_1w - (high_1w - low_1w) * 1.1 / 12.0

    # Align to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)

    # Volume spike: volume > 2.0 * 20-period average (~20 days)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(ema20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Uptrend + breakout above R1 + volume spike
            if close[i] > ema20_1w_aligned[i] and close[i] > r1_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend + breakdown below S1 + volume spike
            elif close[i] < ema20_1w_aligned[i] and close[i] < s1_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or trend turns bearish
            if close[i] < s1_aligned[i] or close[i] < ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or trend turns bullish
            if close[i] > r1_aligned[i] or close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals