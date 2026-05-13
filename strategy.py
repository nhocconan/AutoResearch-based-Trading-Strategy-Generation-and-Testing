#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
# Hypothesis: Camarilla R1/S1 breakout with 1d trend filter and volume spike.
# Long: Close breaks above R1 with volume spike and 1d EMA34 uptrend.
# Short: Close breaks below S1 with volume spike and 1d EMA34 downtrend.
# Exit when price crosses back below R1 (long) or above S1 (short).
# Uses 12h timeframe to reduce trade frequency, targeting 12-37 trades/year.
# Works in bull (buy uptrend) and bear (sell downtrend) via 1d trend filter.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
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
    if len(df_1d) < 50:
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla levels from previous 1d bar
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = np.zeros(n)
    camarilla_s1 = np.zeros(n)
    
    # Use previous day's OHLC (shifted by 1 to avoid look-ahead)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = close_1d[0]  # First value
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    camarilla_r1_1d = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 12
    camarilla_s1_1d = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (wait for 1d bar to close)
    camarilla_r1 = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1 = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    # Get 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if data is not ready
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close breaks above R1 with volume spike and 1d EMA uptrend
            if close[i] > camarilla_r1[i] and volume_spike[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close breaks below S1 with volume spike and 1d EMA downtrend
            elif close[i] < camarilla_s1[i] and volume_spike[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back below R1
            if close[i] < camarilla_r1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back above S1
            if close[i] > camarilla_s1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals