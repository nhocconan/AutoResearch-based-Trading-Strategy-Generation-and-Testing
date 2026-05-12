# 160082: 12h_Donchian_Breakout_1dTrend_Volume
# Hypothesis: Price breaking above/below 12h Donchian channels (20-period) with daily trend filter and volume confirmation.
# Uses 12h timeframe with 1-day EMA trend filter for higher timeframe context. Works in bull/bear by following the daily trend direction.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

#!/usr/bin/env python3
name = "12h_Donchian_Breakout_1dTrend_Volume"
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
    volume = prices['volume'].values

    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')

    # Calculate 1-day EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Calculate 12h Donchian channels (20-period)
    high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Volume confirmation: >1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(35, n):  # Start after EMA34 warmup
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(high_rolling_max[i]) or 
            np.isnan(low_rolling_min[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Donchian upper + EMA34 uptrend + volume confirmation
            if (close[i] > high_rolling_max[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.30
                position = 1
            # SHORT: Price breaks below Donchian lower + EMA34 downtrend + volume confirmation
            elif (close[i] < low_rolling_min[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Donchian lower (trend reversal)
            if close[i] < low_rolling_min[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # EXIT SHORT: Price closes above Donchian upper (trend reversal)
            if close[i] > high_rolling_max[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30

    return signals