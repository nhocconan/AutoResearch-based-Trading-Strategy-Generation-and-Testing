# 6h_WeeklyPivot_PriceAction_Reversal
# Hypothesis: Price reverses at weekly pivot points (R3/S3, R4/S4) with volume confirmation and trend filter. Works in both bull and bear markets by fading extreme levels and capturing breakouts. Weekly pivots act as institutional support/resistance, especially effective during volatile periods.

name = "6h_WeeklyPivot_PriceAction_Reversal"
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

    # Calculate weekly pivot points (based on previous week)
    # Load weekly data once before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly pivot calculation: (H + L + C) / 3
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate pivot and support/resistance levels
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pp - weekly_low
    s1 = 2 * pp - weekly_high
    r2 = pp + (weekly_high - weekly_low)
    s2 = pp - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pp - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pp)
    r4 = r3 + (r3 - r2)
    s4 = s3 - (s2 - s3)
    
    # Align weekly pivot levels to 6h timeframe (with 1-bar delay for weekly close)
    pp_aligned = align_htf_to_ltf(prices, df_weekly, pp)
    r3_aligned = align_htf_to_ltf(prices, df_weekly, r3)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, s3)
    r4_aligned = align_htf_to_ltf(prices, df_weekly, r4)
    s4_aligned = align_htf_to_ltf(prices, df_weekly, s4)
    
    # EMA50 for trend filter (6h timeframe)
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required value is NaN
        if (np.isnan(ema50[i]) or np.isnan(vol_avg_20[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG ENTRY: Price at S3/S4 with rejection + volume + above EMA50
            if ((abs(close[i] - s3_aligned[i]) < 0.001 * close[i] or 
                 abs(close[i] - s4_aligned[i]) < 0.001 * close[i]) and
                volume[i] > vol_avg_20[i] * 1.5 and
                close[i] > ema50[i]):
                signals[i] = 0.25
                position = 1
            # SHORT ENTRY: Price at R3/R4 with rejection + volume + below EMA50
            elif ((abs(close[i] - r3_aligned[i]) < 0.001 * close[i] or 
                   abs(close[i] - r4_aligned[i]) < 0.001 * close[i]) and
                  volume[i] > vol_avg_20[i] * 1.5 and
                  close[i] < ema50[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches pivot point or shows weakness
            if close[i] >= pp_aligned[i] or close[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches pivot point or shows strength
            if close[i] <= pp_aligned[i] or close[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals