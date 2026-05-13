#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: Use 12h Camarilla R3/S3 breakouts with 1w EMA trend filter and volume confirmation.
# Camarilla levels provide reversal points in ranging markets; breakouts above R3 or below S3 capture momentum.
# The 1w EMA filter ensures trades align with the long-term trend, avoiding counter-trend trades in strong trends.
# Works in bull (follows breaks with bullish 1w trend) and bear (avoids bullish breaks in bearish 1w trend).
# Target: 50-150 total trades over 4 years = 12-37/year.

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
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

    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    # Calculate Camarilla levels (based on previous day's OHLC)
    # Typical Camarilla calculation: 
    # R4 = C + ((H-L)*1.5/2)
    # R3 = C + ((H-L)*1.25/2)
    # S3 = C - ((H-L)*1.25/2)
    # S4 = C - ((H-L)*1.5/2)
    # We'll use daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    camarilla_r3 = df_1d['close'] + ((df_1d['high'] - df_1d['low']) * 1.25 / 2)
    camarilla_s3 = df_1d['close'] - ((df_1d['high'] - df_1d['low']) * 1.25 / 2)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_avg_20[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above Camarilla R3 + price above 1w EMA (bullish trend) + volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_34_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Camarilla S3 + price below 1w EMA (bearish trend) + volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_34_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below Camarilla S3 or price below 1w EMA
            if (close[i] < camarilla_s3_aligned[i] or close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above Camarilla R3 or price above 1w EMA
            if (close[i] > camarilla_r3_aligned[i] or close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals