#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Breakout at daily Camarilla R1/S1 levels with 1-day EMA trend filter and volume spike confirmation.
# Uses Camarilla pivot levels (resistance/support) from prior day for institutional breakout levels.
# Filters: price must be above/below 1-day EMA34 for trend alignment, and volume > 1.5x 20-period average.
# Works in bull (breaks above R1 in uptrend) and bear (breaks below S1 in downtrend).
# Target: 20-50 trades per year (~80-200 over 4 years) to stay within fee-efficient range.

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

    # Get 1d data for Camarilla pivots and EMA
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate prior day's Camarilla levels (using previous day's HLC)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We use the prior day's values to avoid look-ahead
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    
    camarilla_r1 = prev_day_close + (prev_day_high - prev_day_low) * 1.1 / 12
    camarilla_s1 = prev_day_close - (prev_day_high - prev_day_low) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (no extra delay needed as pivots are known at day open)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate 1-day EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)

    # Volume filter: >1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above Camarilla R1 + price above 1d EMA (bullish trend) + volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema_34_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Camarilla S1 + price below 1d EMA (bearish trend) + volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema_34_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below Camarilla S1 or price below 1d EMA
            if (close[i] < camarilla_s1_aligned[i] or close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above Camarilla R1 or price above 1d EMA
            if (close[i] > camarilla_r1_aligned[i] or close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals