#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike
Hypothesis: Camarilla pivot levels (R1/S1) breakout with volume confirmation and 1-week EMA50 trend filter on 12h timeframe.
Long: Price breaks above R1 with volume spike and price above 1w EMA50 (uptrend)
Short: Price breaks below S1 with volume spike and price below 1w EMA50 (downtrend)
Exit: Price crosses back through the Camarilla pivot point (mean reversion within the range)
Designed for 15-30 trades/year to minimize fee drag. Uses weekly trend to avoid counter-trend trades in bear markets.
"""

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Camarilla Pivot Levels (based on previous day's range)
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    # Pivot = (high + low + close) / 3
    r1 = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    pivot = np.full(n, np.nan)
    for i in range(1, n):
        # Use previous day's data for today's levels
        if i-1 >= 0:
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            range_val = prev_high - prev_low
            r1[i] = prev_close + range_val * 1.1 / 12
            s1[i] = prev_close - range_val * 1.1 / 12
            pivot[i] = (prev_high + prev_low + prev_close) / 3

    # Volume confirmation: current volume > 2.0 x 24-period average (2 days)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    volume_spike = volume > (2.0 * vol_ma)

    # Get 1-week EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):
        # Skip if data is not ready
        if np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(pivot[i]) or np.isnan(volume_spike[i]) or np.isnan(ema_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above R1 with volume spike and price above 1w EMA50 (uptrend)
            if close[i] > r1[i] and volume_spike[i] and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1 with volume spike and price below 1w EMA50 (downtrend)
            elif close[i] < s1[i] and volume_spike[i] and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below pivot point (mean reversion)
            if close[i] < pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above pivot point
            if close[i] > pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals