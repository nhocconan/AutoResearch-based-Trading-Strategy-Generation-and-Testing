#!/usr/bin/env python3
# 12h_Camarilla_Pivot_Reversal_Trend_Filter
# Hypothesis: Use daily Camarilla pivot levels (S1/S2/S3/S4 for shorts, R1/R2/R3/R4 for longs) with 1d trend filter (EMA34) and volume confirmation (>1.5x 20-period average). Enter long at R3 with bullish 1d trend and volume spike; enter short at S3 with bearish 1d trend and volume spike. Exit when price touches opposite S1/R1 level. Designed for 12h timeframe to avoid overtrading (target: 15-30 trades/year) and work in both bull/bear markets via trend filter.

name = "12h_Camarilla_Pivot_Reversal_Trend_Filter"
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

    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values

    # Calculate Camarilla pivot levels for each day
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), R2 = close + 0.55*(high-low), R1 = close + 0.275*(high-low)
    # S1 = close - 0.275*(high-low), S2 = close - 0.55*(high-low), S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    range_1d = high_1d - low_1d
    r4 = close_1d + 1.5 * range_1d
    r3 = close_1d + 1.1 * range_1d
    r2 = close_1d + 0.55 * range_1d
    r1 = close_1d + 0.275 * range_1d
    s1 = close_1d - 0.275 * range_1d
    s2 = close_1d - 0.55 * range_1d
    s3 = close_1d - 1.1 * range_1d
    s4 = close_1d - 1.5 * range_1d

    # Align Camarilla levels to 12h timeframe (wait for daily close)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3, additional_delay_bars=0)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1, additional_delay_bars=0)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3, additional_delay_bars=0)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1, additional_delay_bars=0)

    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(r3_12h[i]) or np.isnan(r1_12h[i]) or 
            np.isnan(s3_12h[i]) or np.isnan(s1_12h[i]) or
            np.isnan(ema34_12h[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price touches R3 + bullish 1d trend (price > EMA34) + volume spike
            if (close[i] >= r3_12h[i] and 
                close[i] > ema34_12h[i] and
                volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches S3 + bearish 1d trend (price < EMA34) + volume spike
            elif (close[i] <= s3_12h[i] and 
                  close[i] < ema34_12h[i] and
                  volume[i] > vol_avg_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price touches S1 (opposite level)
            if close[i] <= s1_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price touches R1 (opposite level)
            if close[i] >= r1_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals