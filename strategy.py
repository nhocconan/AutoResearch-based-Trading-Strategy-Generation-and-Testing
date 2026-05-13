#!/usr/bin/env python3
"""
1h_Time_Slice_Momentum_With_Volume_Filter
Hypothesis: During active London/NY session overlap (08-16 UTC), price exhibits
persistent momentum that can be captured with a 3-bar EMA crossover.
Trades only in direction of 4h trend (EMA50) to avoid counter-trend whipsaws.
Volume spike confirms institutional participation. Designed for low-frequency,
high-quality setups to minimize fee drag in choppy 1h environment.
Works in bull/bear by following 4h trend direction.
"""
name = "1h_Time_Slice_Momentum_With_Volume_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mplfinance import make_addplot
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']

    # Precompute session hours (08-16 UTC) to reduce noise trades
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours < 16)

    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values

    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)

    # 1h EMA3 and EMA8 for entry timing
    ema3 = pd.Series(close).ewm(span=3, adjust=False, min_periods=3).mean().values
    ema8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values

    # Volume spike: volume > 2.0 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Skip if any required value is NaN
        if np.isnan(ema50_4h_aligned[i]) or np.isnan(ema3[i]) or np.isnan(ema8[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Uptrend + EMA3 crosses above EMA8 + volume spike
            if ema3[i-1] <= ema8[i-1] and ema3[i] > ema8[i] and close[i] > ema50_4h_aligned[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Downtrend + EMA3 crosses below EMA8 + volume spike
            elif ema3[i-1] >= ema8[i-1] and ema3[i] < ema8[i] and close[i] < ema50_4h_aligned[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: EMA3 crosses below EMA8 or trend turns bearish
            if ema3[i] < ema8[i] or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: EMA3 crosses above EMA8 or trend turns bullish
            if ema3[i] > ema8[i] or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals