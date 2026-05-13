#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS
# Hypothesis: Trade breakouts from Camarilla R1/S1 levels on 4h with 12h EMA trend filter and volume confirmation.
# Long when price breaks above R1 in 12h uptrend with volume spike; short when breaks below S1 in 12h downtrend.
# Exit on opposite level break or trend reversal. Designed for 20-30 trades/year to avoid fee drag.
# Works in bull (trend continuation) and bear (mean reversion at extremes) markets.

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS"
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

    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA50 for trend direction
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)

    # Calculate Camarilla levels from previous day (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values  # shift(1) for previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla R1 and S1 levels
    camarilla_r1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    camarilla_s1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 in 12h uptrend with volume spike
            if close[i] > camarilla_r1_aligned[i]:
                if ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1] and volume[i] > vol_avg_20[i] * 1.5:
                    signals[i] = 0.25
                    position = 1
            # SHORT: Price breaks below S1 in 12h downtrend with volume spike
            elif close[i] < camarilla_s1_aligned[i]:
                if ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1] and volume[i] > vol_avg_20[i] * 1.5:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or trend turns down
            if close[i] < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]:  # trend turned down
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or trend turns up
            if close[i] > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]:  # trend turned up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals