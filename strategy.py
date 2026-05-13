#!/usr/bin/env python3
# 6h_Camarilla_R3S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot breakout at R3/S3 levels on 6h, filtered by 1d trend direction and volume spikes.
# Uses 1d OHLC to calculate Camarilla levels for the current day. Trades breakouts in direction of 1d trend (EMA50).
# Volume confirmation requires current volume > 2.0 x 20-period average to filter weak breakouts.
# Designed to work in both bull and bear markets by following 1d trend direction.
# Target: 15-35 trades/year per symbol to minimize fee drag.

name = "6h_Camarilla_R3S3_Breakout_1dTrend_Volume"
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

    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')

    # Calculate Camarilla levels from 1d OHLC: R3, S3, R4, S4
    # Camarilla: Close + (High-Low) * 1.1/4 = R3, Close - (High-Low) * 1.1/4 = S3
    # R4 = Close + (High-Low) * 1.1/2, S4 = Close - (High-Low) * 1.1/2
    hl_range = df_1d['high'] - df_1d['low']
    camarilla_r3 = df_1d['close'] + hl_range * 1.1 / 4
    camarilla_s3 = df_1d['close'] - hl_range * 1.1 / 4
    camarilla_r4 = df_1d['close'] + hl_range * 1.1 / 2
    camarilla_s4 = df_1d['close'] - hl_range * 1.1 / 2

    # Align Camarilla levels to 6h timeframe (wait for 1d bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4.values)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4.values)

    # Trend filter: 1d EMA50
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above R3 in uptrend with volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 in downtrend with volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 or trend turns down
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 or trend turns up
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals