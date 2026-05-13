#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeS
# Hypothesis: Breakouts from Camarilla R3/S3 levels on 4h timeframe, filtered by 1-day trend (EMA34) and volume confirmation.
# Long when price breaks above R3 in uptrend with volume spike.
# Short when price breaks below S3 in downtrend with volume spike.
# Exit on opposite Camarilla level (S3 for long, R3 for short) or trend reversal.
# Uses 1-day EMA34 for trend filter and 4h volume spike (>1.5x 20-bar average).
# Designed for 4h timeframe to achieve ~25-40 trades/year per symbol.
# Works in bull markets via trend-following breakouts and bear via mean-reversion bounces at extreme levels.

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeS"
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

    # Get 1-day data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # 1-day EMA34 for trend direction
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Calculate Camarilla levels from previous day's OHLC
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    # Using previous day's values (shifted by 1)
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels for each day
    camarilla_R3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    camarilla_S3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)

    # Volume filter: >1.5x 20-period average on 4h
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or 
            np.isnan(camarilla_S3_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R3 in uptrend with volume spike
            if close[i] > camarilla_R3_aligned[i]:
                if close[i] > ema_34_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                    signals[i] = 0.25
                    position = 1
            # SHORT: Price breaks below S3 in downtrend with volume spike
            elif close[i] < camarilla_S3_aligned[i]:
                if close[i] < ema_34_1d_aligned[i] and volume[i] > vol_avg_20[i] * 1.5:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 or trend turns down
            if close[i] < camarilla_S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] < ema_34_1d_aligned[i]:  # trend turned down
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 or trend turns up
            if close[i] > camarilla_R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] > ema_34_1d_aligned[i]:  # trend turned up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals