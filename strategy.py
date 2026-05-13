#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Use Camarilla pivot levels (R1/S1) from daily timeframe for breakout entries on 12h, filtered by weekly trend (EMA50) and volume confirmation.
# Camarilla levels provide precise support/resistance based on prior day's range. Breakouts above R1 or below S1 with volume suggest institutional interest.
# Weekly EMA50 filter ensures alignment with longer-term trend, reducing counter-trend trades.
# Volume > 1.5x 20-period average confirms breakout strength.
# Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year).

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
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

    # Get 1d data for Camarilla pivot calculation (based on prior day)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels: based on previous day's high, low, close
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We use shifted values to ensure we only use completed daily data
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Avoid division by zero and handle first row
    hl_range = prev_high - prev_low
    camarilla_r1 = prev_close + hl_range * 1.1 / 12.0
    camarilla_s1 = prev_close - hl_range * 1.1 / 12.0
    
    # Align Camarilla levels to 12h timeframe (wait for daily close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    # Volume filter: >1.5x 20-period average on 12h
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above Camarilla R1 + price above weekly EMA50 (bullish trend) + volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Camarilla S1 + price below weekly EMA50 (bearish trend) + volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below Camarilla S1 or price below weekly EMA50
            if (close[i] < camarilla_s1_aligned[i] or close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above Camarilla R1 or price above weekly EMA50
            if (close[i] > camarilla_r1_aligned[i] or close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals