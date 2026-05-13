#!/usr/bin/env python3
# 1D_Camarilla_R1_S1_WeeklyTrend_Volume
# Hypothesis: Daily Camarilla R1/S1 breakouts with weekly EMA trend filter and volume confirmation.
# Uses Camarilla pivot levels for mean-reversion breakouts in ranging markets, filtered by weekly trend to avoid counter-trend trades.
# Volume spike confirms breakout strength. Designed for low frequency (~10-20 trades/year) to minimize fee drag in 2025 bear market.

name = "1D_Camarilla_R1_S1_WeeklyTrend_Volume"
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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')

    # Calculate Camarilla levels for current day (using previous day's OHLC)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We use previous day's high, low, close to calculate today's levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # First bar uses current high
    prev_low[0] = low[0]
    prev_close[0] = close[0]

    range_prev = prev_high - prev_low
    camarilla_r1 = prev_close + range_prev * 1.1 / 12
    camarilla_s1 = prev_close - range_prev * 1.1 / 12

    # Calculate weekly EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)

    # Volume spike: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(camarilla_r1[i]) or 
            np.isnan(camarilla_s1[i]) or 
            np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above Camarilla R1 with volume spike and weekly uptrend
            if (close[i] > camarilla_r1[i] and 
                volume_spike[i] and 
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Camarilla S1 with volume spike and weekly downtrend
            elif (close[i] < camarilla_s1[i] and 
                  volume_spike[i] and 
                  close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S1 or weekly trend turns down
            if close[i] < camarilla_s1[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R1 or weekly trend turns up
            if close[i] > camarilla_r1[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals