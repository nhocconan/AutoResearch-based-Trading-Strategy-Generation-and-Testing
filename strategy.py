#!/usr/bin/env python3
# 4h_HTF_Trend_LTF_Entry_Volume
# Hypothesis: Use 1d EMA34 as primary trend filter, with 4h Donchian(20) breakout and volume confirmation for entries.
# The strategy takes long positions when price breaks above Donchian upper band in a 1d uptrend with volume spike,
# and short positions when price breaks below Donchian lower band in a 1d downtrend with volume spike.
# Exits occur on Donchian opposite break or trend reversal. This combines trend following with momentum
# breakouts while avoiding whipsaws through volume confirmation and higher timeframe trend alignment.
# Target: 20-50 trades per year (80-200 total over 4 years).

name = "4h_HTF_Trend_LTF_Entry_Volume"
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

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Calculate Donchian Channel (20) on 4h
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values

    # Volume filter: >1.5x 20-period average on 4h
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above Donchian upper + price above 1d EMA34 (uptrend) + volume spike
            if (close[i] > donchian_upper[i] and 
                close[i] > ema_34_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian lower + price below 1d EMA34 (downtrend) + volume spike
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema_34_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below Donchian lower or price below 1d EMA34
            if (close[i] < donchian_lower[i] or close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above Donchian upper or price above 1d EMA34
            if (close[i] > donchian_upper[i] or close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals