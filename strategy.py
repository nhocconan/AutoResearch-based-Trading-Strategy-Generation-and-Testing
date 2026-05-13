#!/usr/bin/env python3
# 6h_WeeklyPivot_MeanReversion_1dTrend
# Hypothesis: Fade extreme weekly pivot levels (R4/S4) with 1-day trend filter and volume confirmation.
# Weekly pivots act as strong support/resistance; reversals at extremes capture mean reversion.
# 1-day trend filter ensures trades align with intermediate trend direction.
# Volume confirmation filters false breakouts. Designed for 15-30 trades/year on 6h timeframe.

name = "6h_WeeklyPivot_MeanReversion_1dTrend"
timeframe = "6h"
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

    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')

    # Calculate weekly pivot points (using prior week's OHLC)
    # Standard pivot: P = (H + L + C) / 3
    # R4 = P + (H - L) * 1.1 * 2
    # S4 = P - (H - L) * 1.1 * 2
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3
    r4 = pivot + (prev_week_high - prev_week_low) * 1.1 * 2
    s4 = pivot - (prev_week_high - prev_week_low) * 1.1 * 2
    
    # Align weekly pivots to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)

    # 1-day EMA34 trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):  # Start after sufficient warmup for EMA34
        # Skip if any required value is NaN
        if (np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price drops to S4 with volume reversal and above daily EMA34
            if (close[i] <= s4_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price rises to R4 with volume reversal and below daily EMA34
            elif (close[i] >= r4_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above weekly pivot or closes below daily EMA34
            if close[i] >= pivot[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses below weekly pivot or closes above daily EMA34
            if close[i] <= pivot[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals