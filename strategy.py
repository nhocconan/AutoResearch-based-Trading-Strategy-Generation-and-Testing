#!/usr/bin/env python3
# 4h_Donchian_Breakout_Volume_Trend
# Hypothesis: Donchian(20) breakouts with volume confirmation and EMA50 trend filter capture strong momentum moves.
# Uses 4h timeframe with 1d EMA34 for higher timeframe trend confirmation.
# Entry: Long when price breaks above Donchian upper + volume spike + EMA50 up + EMA34(1d) up.
#         Short when price breaks below Donchian lower + volume spike + EMA50 down + EMA34(1d) down.
# Exit: Mean reversion to Donchian middle (20-period midpoint) to avoid overstaying.
# Target: 20-35 trades/year on 4h to stay within optimal range while capturing significant moves.

name = "4h_Donchian_Breakout_Volume_Trend"
timeframe = "4h"
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

    # Donchian Channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0

    # EMA50 for trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values

    # Volume confirmation: volume > 2.0x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    # Higher timeframe trend: 1d EMA34
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema50[i]) or np.isnan(vol_avg_20[i]) or np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above upper + volume spike + EMA50 up + EMA34(1d) up
            if (close[i] > highest_high[i] and 
                volume[i] > vol_avg_20[i] * 2.0 and
                close[i] > ema50[i] and
                ema34_1d_aligned[i] > ema34_1d_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below lower + volume spike + EMA50 down + EMA34(1d) down
            elif (close[i] < lowest_low[i] and 
                  volume[i] > vol_avg_20[i] * 2.0 and
                  close[i] < ema50[i] and
                  ema34_1d_aligned[i] < ema34_1d_aligned[i-1]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Mean reversion to Donchian middle
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Mean reversion to Donchian middle
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals