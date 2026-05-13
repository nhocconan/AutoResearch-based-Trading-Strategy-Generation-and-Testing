#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Combine Camarilla pivot levels (R3/S3) with 1d trend filter and volume confirmation on the 12h timeframe.
# Camarilla levels provide high-probability reversal points. We go long when price breaks above R3 with volume and 1d uptrend,
# and short when price breaks below S3 with volume and 1d downtrend. Exits occur when price returns to the Camarilla mid-point (C).
# This strategy targets low-frequency, high-conviction trades suitable for 12h timeframe, avoiding overtrading.
# Expected trades: 50-150 over 4 years (12-37/year).

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)

    # Calculate Camarilla levels from previous day
    # Typical Camarilla calculation uses previous day's high, low, close
    # We'll use daily data to calculate levels, then align to 12h
    prev_high = df_1d['high'].shift(1).values  # Previous day's high
    prev_low = df_1d['low'].shift(1).values    # Previous day's low
    prev_close = df_1d['close'].shift(1).values # Previous day's close
    
    # Calculate Camarilla levels
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    # C = close (pivot point)
    rng = prev_high - prev_low
    camarilla_r3 = prev_close + rng * 1.1 / 2.0
    camarilla_s3 = prev_close - rng * 1.1 / 2.0
    camarilla_c = prev_close  # Pivot point
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_c_aligned = align_htf_to_ltf(prices, df_1d, camarilla_c)

    # Volume filter: >1.5x 20-period average on 12h
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):
        # Skip if any required value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_c_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above R3 + volume spike + price above 1d EMA50 (bullish trend)
            if (close[i] > camarilla_r3_aligned[i] and 
                volume[i] > vol_avg_20[i] * 1.5 and
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S3 + volume spike + price below 1d EMA50 (bearish trend)
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume[i] > vol_avg_20[i] * 1.5 and
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to or below Camarilla C (pivot point)
            if close[i] <= camarilla_c_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to or above Camarilla C (pivot point)
            if close[i] >= camarilla_c_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals