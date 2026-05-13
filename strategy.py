#!/usr/bin/env python3
# 1d_Donchian_Breakout_1wTrend_Volume
# Hypothesis: Use daily Donchian channel (20-day) breakouts with weekly trend filter and volume confirmation.
# Enter long when price breaks above 20-day high with weekly EMA uptrend and volume spike.
# Enter short when price breaks below 20-day low with weekly EMA downtrend and volume spike.
# Exit when price returns to the 20-day midpoint to avoid reversals.
# This strategy captures strong trending moves while filtering false breakouts with weekly trend and volume.
# Target: 20-30 trades/year on 1d to minimize fee decay while capturing strong moves.

name = "1d_Donchian_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    close_1w = df_1w['close'].values

    # Calculate Donchian channel (20-day) from daily data
    # Using rolling window on daily prices
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max_20 + low_min_20) / 2.0

    # Calculate weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)

    # Align Donchian levels to daily timeframe (already daily, but ensure alignment)
    # Note: Donchian is calculated on daily data, so no alignment needed for same timeframe
    # But we'll keep the structure for consistency

    # Volume confirmation: volume > 2.0x 20-period average (to filter weak moves)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above 20-day high + weekly EMA uptrend + volume spike
            if (close[i] > high_max_20[i] and 
                close[i] > ema34_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below 20-day low + weekly EMA downtrend + volume spike
            elif (close[i] < low_min_20[i] and 
                  close[i] < ema34_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses back below 20-day midpoint
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses back above 20-day midpoint
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals