#!/usr/bin/env python3
# 1d_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Use weekly trend via 200-period EMA (aligned to daily) as primary filter, then take Camarilla R1/S1 breakouts on the daily chart with volume confirmation. Weekly EMA200 ensures we only trade in the direction of the long-term trend, reducing false signals in ranging markets. Camarilla levels provide precise intraday support/resistance, and volume confirms breakout strength. Designed to work in both bull and bear markets by aligning with the weekly trend, with tight entry conditions to limit trades and minimize fee drag.

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

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 200-period EMA on weekly close for trend filter
    ema_200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)

    # Calculate Camarilla levels for the current day using previous day's OHLC
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We shift by 1 to use previous day's data
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # Set first value to NaN since no prior day
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    rang = prev_high - prev_low
    camarilla_r1 = prev_close + rang * 1.1 / 12
    camarilla_s1 = prev_close - rang * 1.1 / 12

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above Camarilla R1 + price above weekly EMA200 (uptrend) + volume spike
            if (close[i] > camarilla_r1[i] and 
                close[i] > ema_200_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Camarilla S1 + price below weekly EMA200 (downtrend) + volume spike
            elif (close[i] < camarilla_s1[i] and 
                  close[i] < ema_200_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below Camarilla S1 or price below weekly EMA200
            if (close[i] < camarilla_s1[i] or close[i] < ema_200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above Camarilla R1 or price above weekly EMA200
            if (close[i] > camarilla_r1[i] or close[i] > ema_200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals