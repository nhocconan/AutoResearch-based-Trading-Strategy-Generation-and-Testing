#!/usr/bin/env python3
"""
1d_Pivot_Reversal_with_1w_Trend_Filter
Hypothesis: Daily reversals at weekly key levels (weekly high/low) with volume confirmation 
and weekly trend filter work in both bull and bear markets. Uses weekly high/low as dynamic 
support/resistance, entering reversals when price rejects these levels with volume spike.
Weekly trend filter ensures we only take reversals against the weekly trend (counter-trend 
in strong trends, but only when exhaustion is likely). Designed for 10-25 trades/year.
"""

name = "1d_Pivot_Reversal_with_1w_Trend_Filter"
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

    # Get weekly data (call once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)

    # Calculate weekly high and low
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values

    # Weekly trend: EMA(5) vs EMA(20) on weekly close
    ema5_1w = pd.Series(weekly_close).ewm(span=5, adjust=False, min_periods=5).mean().values
    ema20_1w = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_uptrend = ema5_1w > ema20_1w  # True when weekly uptrend

    # Align weekly levels to daily
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))

    # Daily volume spike: volume > 1.5x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_avg_20 * 1.5)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if weekly data not yet aligned
        if np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        wh = weekly_high_aligned[i]
        wl = weekly_low_aligned[i]
        w_up = weekly_uptrend_aligned[i] > 0.5
        vol_spike = volume_spike[i]

        if position == 0:
            # LONG: Price rejects weekly low with volume spike in weekly downtrend
            # (expecting bounce in downtrend)
            if close[i] <= wl * 1.002 and not w_up and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Price rejects weekly high with volume spike in weekly uptrend
            # (expecting rejection in uptrend)
            elif close[i] >= wh * 0.998 and w_up and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches weekly high or loses momentum
            if close[i] >= wh * 0.995 or (close[i] < close[i-1] and not vol_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches weekly low or loses momentum
            if close[i] <= wl * 1.005 or (close[i] > close[i-1] and not vol_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals