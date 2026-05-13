#!/usr/bin/env python3
# 12h_34EMA_Momentum_1wTrend_Volume
# Hypothesis: 12h EMA34 momentum with weekly trend filter and volume confirmation.
# In bull markets, price above 12h EMA34 captures momentum; weekly trend filter avoids counter-trend trades.
# In bear markets, the weekly trend filter prevents longs in downtrends and shorts in uptrends.
# Volume confirmation ensures breakouts have participation. Target: 50-150 total trades over 4 years.

name = "12h_34EMA_Momentum_1wTrend_Volume"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)

    # Volume filter: >1.5x 20-period average on 12h
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    # Calculate 12h EMA34 for momentum
    ema_34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_avg_20[i]) or 
            np.isnan(ema_34[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price above 12h EMA34 + weekly uptrend + volume spike
            if (close[i] > ema_34[i] and 
                ema_34_1w_aligned[i] > ema_34_1w_aligned[max(0, i-1)] and  # weekly EMA rising
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price below 12h EMA34 + weekly downtrend + volume spike
            elif (close[i] < ema_34[i] and 
                  ema_34_1w_aligned[i] < ema_34_1w_aligned[max(0, i-1)] and  # weekly EMA falling
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price below 12h EMA34 or weekly trend turns down
            if (close[i] < ema_34[i] or 
                ema_34_1w_aligned[i] < ema_34_1w_aligned[max(0, i-1)]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price above 12h EMA34 or weekly trend turns up
            if (close[i] > ema_34[i] or 
                ema_34_1w_aligned[i] > ema_34_1w_aligned[max(0, i-1)]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals