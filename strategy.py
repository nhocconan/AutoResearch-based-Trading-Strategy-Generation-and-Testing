#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot R3/S3 breakouts on 12h timeframe with 1d trend filter (EMA34) and volume spike confirmation.
# Camarilla levels provide high-probability reversal/breakout zones; 1d EMA ensures alignment with daily trend.
# Volume spike confirms institutional participation. Designed for 12-25 trades/year to minimize fee drag.
# Works in bull/bear: long when price breaks above R3 with 1d uptrend; short when breaks below S3 with 1d downtrend.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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

    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')

    # Calculate Camarilla levels from previous day's OHLC
    # R4 = close + (high - low) * 1.5
    # R3 = close + (high - low) * 1.25
    # S3 = close - (high - low) * 1.25
    # S4 = close - (high - low) * 1.5
    # We use R3 and S3 as breakout levels
    camarilla_r3 = df_1d['close'] + (df_1d['high'] - df_1d['low']) * 1.25
    camarilla_s3 = df_1d['close'] - (df_1d['high'] - df_1d['low']) * 1.25
    
    # Align Camarilla levels to 12h timeframe (previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)

    # 1d EMA34 trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):  # Start after sufficient warmup for EMA34
        # Skip if any required value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above Camarilla R3, above 1d EMA34, with volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3, below 1d EMA34, with volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S3 or below 1d EMA34
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R3 or above 1d EMA34
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals