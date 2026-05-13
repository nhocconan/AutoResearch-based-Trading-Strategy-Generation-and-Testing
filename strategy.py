#!/usr/bin/env python3
# 1d_Weekly_Camarilla_R1S1_Breakout_Trend_Filter
# Hypothesis: Enter long when price breaks above weekly Camarilla R1 with price above 200-week EMA (bullish trend) and volume > 1.5x average.
# Enter short when price breaks below weekly Camarilla S1 with price below 200-week EMA (bearish trend) and volume > 1.5x average.
# Uses weekly Camarilla levels for key support/resistance, 200-week EMA for trend filter, and volume surge for confirmation.
# Designed for low frequency to avoid fee drag, works in both bull (breakouts above R1 in uptrend) and bear (breakdowns below S1 in downtrend).

name = "1d_Weekly_Camarilla_R1S1_Breakout_Trend_Filter"
timeframe = "1d"
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

    # Get weekly data for Camarilla levels and trend
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla levels
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # Camarilla levels calculation
    R1 = close_w + 1.1 * (high_w - low_w) / 12
    S1 = close_w - 1.1 * (high_w - low_w) / 12
    
    # Weekly trend: EMA200
    ema200_1w = pd.Series(close_w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly indicators to daily timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(ema200_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Close > R1 + price above 200-week EMA + volume confirmation
            if close[i] > R1_aligned[i] and close[i] > ema200_1w_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close < S1 + price below 200-week EMA + volume confirmation
            elif close[i] < S1_aligned[i] and close[i] < ema200_1w_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below S1 or trend reversal (price below 200-week EMA)
            if close[i] < S1_aligned[i] or close[i] < ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above R1 or trend reversal (price above 200-week EMA)
            if close[i] > R1_aligned[i] or close[i] > ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals