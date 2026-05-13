#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: 12h Camarilla R1/S1 breakout with volume confirmation and 1d EMA50 trend filter. 
# Long when price breaks above R1 with volume spike and price above 1d EMA50 (uptrend).
# Short when price breaks below S1 with volume spike and price below 1d EMA50 (downtrend).
# Exit when price crosses back through the Camarilla Pivot Point (mean reversion within the range).
# Uses 12h timeframe to reduce trade frequency and avoid fee drag. Works in both bull and bear by capturing breakouts with trend alignment.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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

    # Calculate previous day's Camarilla levels (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    R1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12
    S1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12
    PP_1d = (high_1d + low_1d + close_1d) / 3  # Pivot Point
    
    # Align to 12h timeframe (wait for 1d candle to close)
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    PP_1d_aligned = align_htf_to_ltf(prices, df_1d, PP_1d)
    
    # Volume confirmation: current volume > 2.0 x 24-period average (24 * 12h = 12 days)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    volume_spike = volume > (2.0 * vol_ma)

    # Get 1d EMA50 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(24, n):
        # Skip if data is not ready
        if (np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]) or 
            np.isnan(PP_1d_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(ema_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: break above R1 with volume spike and price above 1d EMA50 (uptrend)
            if close[i] > R1_1d_aligned[i] and volume_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1 with volume spike and price below 1d EMA50 (downtrend)
            elif close[i] < S1_1d_aligned[i] and volume_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below Pivot Point (mean reversion)
            if close[i] < PP_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above Pivot Point
            if close[i] > PP_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals