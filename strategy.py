#!/usr/bin/env python3
# 6h_Williams_R_Extreme_1dTrend_Volume
# Hypothesis: Williams %R extremes indicate overbought/oversold conditions that precede reversals.
# In bull markets, we take long entries when Williams %R < -80 (oversold) and price > 1d EMA34.
# In bear markets, we take short entries when Williams %R > -20 (overbought) and price < 1d EMA34.
# The 1d EMA34 filter ensures alignment with the daily trend, reducing false signals.
# Volume confirmation (>1.5x 20-period average) adds conviction to reversal setups.
# Williams %R is calculated as (Highest High - Close) / (Highest High - Lowest Low) * -100 over 14 periods.
# Target: 50-150 total trades over 4 years = 12-37/year.

name = "6h_Williams_R_Extreme_1dTrend_Volume"
timeframe = "6h"
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

    # Calculate Williams %R (14) on 6h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)

    # Volume filter: >1.5x 20-period average on 6h
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Williams %R < -80 (oversold) + price above 1d EMA34 (bullish trend) + volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema_34_1d_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) + price below 1d EMA34 (bearish trend) + volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema_34_1d_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R > -20 (overbought) or price below 1d EMA34
            if (williams_r[i] > -20 or close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R < -80 (oversold) or price above 1d EMA34
            if (williams_r[i] < -80 or close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals