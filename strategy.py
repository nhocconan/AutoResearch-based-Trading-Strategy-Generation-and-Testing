#!/usr/bin/env python3

# 1d_1W_Volume_Weighted_Camarilla_Breakout
# Hypothesis: Use weekly Camarilla pivot levels (R1/S1) on 1d timeframe with volume confirmation.
# Long when price breaks above weekly R1 with volume surge; short when breaks below weekly S1 with volume surge.
# Weekly pivots provide structural support/resistance that works in both bull and bear markets.
# Volume confirmation filters out false breakouts. Targets 10-20 trades/year to minimize fee drag.

name = "1d_1W_Volume_Weighted_Camarilla_Breakout"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)

    # Calculate weekly Camarilla levels (R1, S1)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_range = weekly_high - weekly_low
    camarilla_r1 = weekly_close + (weekly_range * 1.1 / 12)
    camarilla_s1 = weekly_close - (weekly_range * 1.1 / 12)
    
    # Align weekly Camarilla levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)

    # Volume confirmation: current volume > 2.0x average of last 20 days
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Break above weekly R1 with volume confirmation
            if close[i] > r1_aligned[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below weekly S1 with volume confirmation
            elif close[i] < s1_aligned[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below weekly S1 (mean reversion to pivot)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above weekly R1 (mean reversion to pivot)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals