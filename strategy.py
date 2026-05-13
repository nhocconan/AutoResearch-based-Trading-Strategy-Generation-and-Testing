#!/usr/bin/env python3
# 4h_Camarilla_Pivot_Reversal_With_Volume_Filter
# Hypothesis: Camarilla pivot levels (especially S1/S3 and R1/R3) act as strong support/resistance in mean-reverting markets.
# In BTC/ETH, price often reverses near these levels during ranging or weak trending periods.
# Strategy: Enter long at S1/S3 with volume confirmation; enter short at R1/R3 with volume confirmation.
# Use 1d timeframe for pivot calculation and 4h for execution.
# Add 1d EMA50 trend filter to avoid counter-trend trades in strong trends.
# Target low trade frequency (<30/year) to minimize fee drag and improve generalization.

name = "4h_Camarilla_Pivot_Reversal_With_Volume_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Calculate Camarilla levels from 1d OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    # Use previous day's OHLC for today's Camarilla levels (no look-ahead)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values

    # Calculate Camarilla levels
    R4 = prev_close + ((prev_high - prev_low) * 1.5000)
    R3 = prev_close + ((prev_high - prev_low) * 1.2500)
    R2 = prev_close + ((prev_high - prev_low) * 1.1666)
    R1 = prev_close + ((prev_high - prev_low) * 1.0833)
    S1 = prev_close - ((prev_high - prev_low) * 1.0833)
    S2 = prev_close - ((prev_high - prev_low) * 1.1666)
    S3 = prev_close - ((prev_high - prev_low) * 1.2500)
    S4 = prev_close - ((prev_high - prev_low) * 1.5000)

    # Align 1d Camarilla levels to 4h timeframe (wait for 1d bar to close)
    R1_1d = align_htf_to_ltf(prices, df_1d, R1)
    R3_1d = align_htf_to_ltf(prices, df_1d, R3)
    S1_1d = align_htf_to_ltf(prices, df_1d, S1)
    S3_1d = align_htf_to_ltf(prices, df_1d, S3)

    # 1d EMA50 trend filter (avoid counter-trend trades)
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume filter: >1.8x 24-period average (6 hours worth of 4h bars)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(60, n):
        # Skip if any required value is NaN
        if (np.isnan(R1_1d[i]) or np.isnan(R3_1d[i]) or np.isnan(S1_1d[i]) or np.isnan(S3_1d[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price at S1 or S3 with volume confirmation AND above 1d EMA50 (bullish bias)
            if ((abs(close[i] - S1_1d[i]) < 0.001 * close[i] or abs(close[i] - S3_1d[i]) < 0.001 * close[i]) and
                volume[i] > vol_avg_24[i] * 1.8 and
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price at R1 or R3 with volume confirmation AND below 1d EMA50 (bearish bias)
            elif ((abs(close[i] - R1_1d[i]) < 0.001 * close[i] or abs(close[i] - R3_1d[i]) < 0.001 * close[i]) and
                  volume[i] > vol_avg_24[i] * 1.8 and
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches R1 (profit target) or breaks below S1 (stop)
            if (close[i] >= R1_1d[i] or close[i] <= S1_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches S1 (profit target) or breaks above R1 (stop)
            if (close[i] <= S1_1d[i] or close[i] >= R1_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals