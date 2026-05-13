#!/usr/bin/env python3
# 1d_Donchian_Breakout_20_1wTrend_Volume
# Hypothesis: Use daily Donchian channel breakouts with 1-week trend filter and volume confirmation.
# Enter long when price breaks above 20-day high with volume spike and 1-week EMA uptrend.
# Enter short when price breaks below 20-day low with volume spike and 1-week EMA downtrend.
# Exit when price returns to the 20-day midpoint (mean reversion).
# Designed to work in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend).
# Target: 10-25 trades/year per symbol.

name = "1d_Donchian_Breakout_20_1wTrend_Volume"
timeframe = "1d"
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

    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)

    close_1w = df_1w['close'].values

    # Calculate 20-day Donchian channels
    high_20 = np.full(n, np.nan)
    low_20 = np.full(n, np.nan)
    for i in range(20, n):
        high_20[i] = np.max(high[i-20:i])
        low_20[i] = np.min(low[i-20:i])
    donchian_mid = (high_20 + low_20) / 2.0

    # Volume confirmation: current volume > 1.5 x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    # Get 1-week EMA34 for trend filter
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(volume_spike[i]) or
            np.isnan(ema_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above 20-day high with volume spike and 1w EMA uptrend
            if close[i] > high_20[i] and volume_spike[i] and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below 20-day low with volume spike and 1w EMA downtrend
            elif close[i] < low_20[i] and volume_spike[i] and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to 20-day midpoint (mean reversion)
            if close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to 20-day midpoint (mean reversion)
            if close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals