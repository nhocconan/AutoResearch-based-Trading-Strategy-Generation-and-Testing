#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Trade Camarilla pivot breakouts on 12-hour timeframe with daily trend filter and volume confirmation.
# Long when price breaks above R3 level during 1-day uptrend with volume spike.
# Short when price breaks below S3 level during 1-day downtrend with volume spike.
# Uses 1-day trend filter to avoid counter-trend whipsaws and volume confirmation to ensure momentum.
# Designed for 12-37 trades per year (50-150 total over 4 years) to minimize fee drag.
# Works in bull markets (trend-following breakouts) and bear markets (trend-following breakdowns).

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

    # Get 1d data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # 1-day EMA34 for trend direction
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1-day OHLC
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # Using previous day's values to avoid look-ahead
    phigh = df_1d['high'].values
    plow = df_1d['low'].values
    pclose = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1-day bar
    R3 = pclose + 1.1 * (phigh - plow)
    S3 = pclose - 1.1 * (phigh - plow)
    
    # Align Camarilla levels to 12h timeframe (wait for 1-day close)
    R3_12h = align_htf_to_ltf(prices, df_1d, R3)
    S3_12h = align_htf_to_ltf(prices, df_1d, S3)

    # Volume filter: >1.5x 30-period average
    vol_avg_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start after warmup period
        # Skip if any required value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(R3_12h[i]) or 
            np.isnan(S3_12h[i]) or np.isnan(vol_avg_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R3 during 1-day uptrend with volume spike
            if close[i] > R3_12h[i] and close[i-1] <= R3_12h[i-1]:  # Breakout above R3
                if ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and volume[i] > vol_avg_30[i] * 1.5:
                    signals[i] = 0.25
                    position = 1
            # SHORT: Price breaks below S3 during 1-day downtrend with volume spike
            elif close[i] < S3_12h[i] and close[i-1] >= S3_12h[i-1]:  # Breakdown below S3
                if ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and volume[i] > vol_avg_30[i] * 1.5:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 or trend turns down
            if close[i] < S3_12h[i] and close[i-1] >= S3_12h[i-1]:  # Breakdown below S3
                signals[i] = 0.0
                position = 0
            elif ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:  # Trend turned down
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 or trend turns up
            if close[i] > R3_12h[i] and close[i-1] <= R3_12h[i-1]:  # Breakout above R3
                signals[i] = 0.0
                position = 0
            elif ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:  # Trend turned up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals