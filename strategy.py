#/usr/bin/env python3
# 4h_EqualHighsLows_RangeBreakout_1dTrend_VolumeFilter
# Hypothesis: In BTC/ETH, range-bound markets (equally spaced highs/lows) precede breakouts.
# We detect a 4-bar range (equal highs/lows ±0.15%) and breakout in direction of 1d EMA50 with volume spike.
# Works in bull/bear: ranges form before major moves in both regimes. Low trade frequency avoids fee drag.
# Target: 20-30 trades/year (80-120 total over 4 years).

name = "4h_EqualHighsLows_RangeBreakout_1dTrend_VolumeFilter"
timeframe = "4h"
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

    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)

    close_1d = df_1d['close'].values

    # Calculate daily EMA50 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    # Detect 4-bar range: equal highs and lows within 0.15%
    range_high = np.full(n, np.nan)
    range_low = np.full(n, np.nan)
    in_range = np.zeros(n, dtype=bool)

    for i in range(4, n):
        hh = np.max(high[i-4:i])
        ll = np.min(low[i-4:i])
        # Check if highs and lows are roughly equal (within 0.15%)
        if (hh - np.min(high[i-4:i])) / hh < 0.0015 and (np.max(low[i-4:i]) - ll) / ll < 0.0015:
            range_high[i] = hh
            range_low[i] = ll
            in_range[i] = True

    # Volume confirmation: current volume > 2.0 x 20-period average (strict)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(range_high[i]) or np.isnan(range_low[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # Only enter on breakout AFTER a range is identified (lookback 1 bar)
            if in_range[i-1]:
                # LONG: Break above range high with volume spike and daily uptrend
                if close[i] > range_high[i-1] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # SHORT: Break below range low with volume spike and daily downtrend
                elif close[i] < range_low[i-1] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters range or daily trend turns down
            if close[i] < range_low[i-1] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters range or daily trend turns up
            if close[i] > range_high[i-1] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals