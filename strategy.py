#!/usr/bin/env python3
# 1d_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Use 1d Camarilla R1/S1 level breakouts with 1w EMA trend filter and volume confirmation.
# Camarilla levels provide high-probability reversal/breakout points. 1w EMA ensures alignment with weekly trend.
# Works in bull (buys R1 breakout in uptrend) and bear (sells S1 breakdown in downtrend).
# Target: 30-100 total trades over 4 years = 7-25/year.

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "1d"
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

    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)

    # Calculate Camarilla levels from previous day
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    camarilla_r1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    camarilla_s1 = prev_close - 1.1 * (prev_high - prev_low) / 12

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_avg_20[i]) or 
            np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above R1 + price above 1w EMA (bullish trend) + volume spike
            if (close[i] > camarilla_r1[i] and 
                close[i] > ema_20_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S1 + price below 1w EMA (bearish trend) + volume spike
            elif (close[i] < camarilla_s1[i] and 
                  close[i] < ema_20_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below S1 or price below 1w EMA
            if (close[i] < camarilla_s1[i] or close[i] < ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above R1 or price above 1w EMA
            if (close[i] > camarilla_r1[i] or close[i] > ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals