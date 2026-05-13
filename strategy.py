#!/usr/bin/env python3
# 12h_Camarilla_R3S3_Breakout_1dTrend_Volume
# Hypothesis: Breakout of Camarilla R3/S3 levels on 12h timeframe, filtered by 1d trend direction and volume confirmation.
# Long when price breaks above R3 during 1d uptrend with volume spike.
# Short when price breaks below S3 during 1d downtrend with volume spike.
# Exit on reversal to R4/S4 or trend reversal.
# Uses 1d trend filter to avoid counter-trend whipsaws, targeting 15-35 trades/year per symbol.
# Designed to work in both bull (trend-following breakouts) and bear (mean-reversion bounces) markets.

name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
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

    # Get 1d data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous 1d bar
    # R3 = close + 1.1*(high - low)/2, S3 = close - 1.1*(high - low)/2
    # R4 = close + 1.1*(high - low), S4 = close - 1.1*(high - low)
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    camarilla_r3 = c_1d + 1.1 * (h_1d - l_1d) / 2
    camarilla_s3 = c_1d - 1.1 * (h_1d - l_1d) / 2
    camarilla_r4 = c_1d + 1.1 * (h_1d - l_1d)
    camarilla_s4 = c_1d - 1.1 * (h_1d - l_1d)
    
    # Align Camarilla levels to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_12h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_12h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # 1d EMA34 for trend direction
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(r4_12h[i]) or np.isnan(s4_12h[i]) or
            np.isnan(ema_34_12h[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above R3 in 1d uptrend with volume spike
            if close[i] > r3_12h[i]:
                if close[i] > ema_34_12h[i] and volume[i] > vol_avg_20[i] * 1.5:
                    signals[i] = 0.25
                    position = 1
            # SHORT: Break below S3 in 1d downtrend with volume spike
            elif close[i] < s3_12h[i]:
                if close[i] < ema_34_12h[i] and volume[i] > vol_avg_20[i] * 1.5:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Break below R4 or trend turns down
            if close[i] < r4_12h[i] or close[i] < ema_34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Break above S4 or trend turns up
            if close[i] > s4_12h[i] or close[i] > ema_34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals