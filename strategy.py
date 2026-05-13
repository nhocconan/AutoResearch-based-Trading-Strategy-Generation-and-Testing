#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot levels (R1/S1) from daily timeframe provide key support/resistance levels.
# Breaking above R1 with daily uptrend and volume surge indicates strong bullish momentum.
# Breaking below S1 with daily downtrend and volume surge indicates strong bearish momentum.
# Exit when price reverts to the daily pivot point (mean reversion to equilibrium).
# Uses only daily timeframe for context to avoid overtrading.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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

    # Get daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from daily OHLC
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    #          R2 = close + 0.6*(high-low), R1 = close + 0.4*(high-low)
    #          S1 = close - 0.4*(high-low), S2 = close - 0.6*(high-low)
    #          S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    daily_range = df_1d['high'].values - df_1d['low'].values
    camarilla_r1 = df_1d['close'].values + 0.4 * daily_range
    camarilla_s1 = df_1d['close'].values - 0.4 * daily_range
    camarilla_pivot = df_1d['close'].values  # Using close as pivot point
    
    # Align Camarilla levels to 4h timeframe (wait for daily close)
    r1_4h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    pivot_4h = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: volume > 2.0x 24-period average (6 hours)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):
        # Skip if any required value is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(pivot_4h[i]) or np.isnan(ema34_4h[i]) or 
            np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close above R1 + daily uptrend + volume spike
            if (close[i] > r1_4h[i] and 
                close[i] > ema34_4h[i] and
                volume[i] > vol_avg_24[i] * 2.0):
                signals[i] = 0.25
                position = 1
            # SHORT: Close below S1 + daily downtrend + volume spike
            elif (close[i] < s1_4h[i] and 
                  close[i] < ema34_4h[i] and
                  volume[i] > vol_avg_24[i] * 2.0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Mean reversion to daily pivot
            if close[i] < pivot_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Mean reversion to daily pivot
            if close[i] > pivot_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals