#!/usr/bin/env python3
# 12h_Camarilla_Pivot_Signal_WeeklyTrend_Volume
# Hypothesis: Use Camarilla pivot levels from daily data for entry signals in the direction of weekly trend, confirmed by volume spikes.
# Camarilla levels (R1,R2,R3,S1,S2,S3) act as intraday support/resistance with high probability of reversal/continuation.
# Weekly trend filter ensures alignment with higher timeframe momentum, reducing false signals.
# Volume spike confirms institutional participation. Works in bull/bear as it captures mean reversion at key levels.
# Low frequency due to requirement of weekly trend alignment and volume confirmation.

name = "12h_Camarilla_Pivot_Signal_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')

    # Calculate Camarilla levels from previous day's OHLC
    # Formula: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), R2 = C + ((H-L) * 1.1/6), R1 = C + ((H-L) * 1.1/12)
    #        S1 = C - ((H-L) * 1.1/12), S2 = C - ((H-L) * 1.1/6), S3 = C - ((H-L) * 1.1/4), S4 = C - ((H-L) * 1.1/2)
    # Where C = (H+L+C)/3 (typical price), but we use close of previous day for pivot
    # Actually standard Camarilla uses (H+L+C)/3 as pivot, but we'll use typical price
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    high_val = df_1d['high'].values
    low_val = df_1d['low'].values
    close_val = df_1d['close'].values

    pivot = typical_price.values
    range_hl = high_val - low_val

    R1 = pivot + (range_hl * 1.1 / 12)
    R2 = pivot + (range_hl * 1.1 / 6)
    R3 = pivot + (range_hl * 1.1 / 4)
    S1 = pivot - (range_hl * 1.1 / 12)
    S2 = pivot - (range_hl * 1.1 / 6)
    S3 = pivot - (range_hl * 1.1 / 4)

    # Align Camarilla levels to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)

    # Weekly trend: EMA50 on weekly close
    weekly_close = df_1w['close'].values
    ema50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Volume spike: volume > 2.0 * 4-period average (2 days worth at 12h)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > 2.0 * vol_ma_4

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(R2_aligned[i]) or np.isnan(R3_aligned[i]) or
            np.isnan(S1_aligned[i]) or np.isnan(S2_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price crosses above S1 with weekly uptrend and volume spike
            if close[i] > S1_aligned[i] and close[i-1] <= S1_aligned[i-1] and ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below R1 with weekly downtrend and volume spike
            elif close[i] < R1_aligned[i] and close[i-1] >= R1_aligned[i-1] and ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S2 or weekly trend turns down
            if close[i] < S2_aligned[i] and close[i-1] >= S2_aligned[i-1] or ema50_1w_aligned[i] < ema50_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above R2 or weekly trend turns up
            if close[i] > R2_aligned[i] and close[i-1] <= R2_aligned[i-1] or ema50_1w_aligned[i] > ema50_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals