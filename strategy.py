#/usr/bin/env python3
# 12h_WilliamsAlligator_Trend_Confirmation
# Hypothesis: Williams Alligator identifies trend direction and strength through smoothed moving averages (Jaw, Teeth, Lips).
# Combines with 1-week trend filter and volume confirmation to capture trends in both bull and bear markets.
# Designed for 12h timeframe to target 12-37 trades per year, minimizing fee drag.
# Williams Alligator is effective in trending markets and avoids whipsaws during consolidation.

name = "12h_WilliamsAlligator_Trend_Confirmation"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(arr, period):
    """Smoothed Moving Average (SMMA)"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    result = np.full_like(arr, np.nan, dtype=float)
    sma = np.nansum(arr[:period]) / period
    result[period-1] = sma
    for i in range(period, len(arr)):
        result[i] = (result[i-1] * (period-1) + arr[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)

    close_1w = df_1w['close'].values

    # Calculate Williams Alligator on 12h data
    # Jaw (13-period SMMA, shifted 8 bars forward)
    # Teeth (8-period SMMA, shifted 5 bars forward)
    # Lips (5-period SMMA, shifted 3 bars forward)
    median_price = (high + low) / 2
    
    jaw_raw = smma(median_price, 13)
    teeth_raw = smma(median_price, 8)
    lips_raw = smma(median_price, 5)
    
    # Apply shifts (Williams Alligator specific)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Set initial values to NaN to avoid roll artifacts
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan

    # Get weekly EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)

    # Volume confirmation: 1.5x 20-period SMA
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = volume_sma20 * 1.5

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(13, n):  # Start after Alligator needs 13 bars
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Lips > Teeth > Jaw (bullish alignment) with volume confirmation and weekly uptrend
            if (lips[i] > teeth[i] > jaw[i] and
                volume[i] > volume_threshold[i] and
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Jaw > Teeth > Lips (bearish alignment) with volume confirmation and weekly downtrend
            elif (jaw[i] > teeth[i] > lips[i] and
                  volume[i] > volume_threshold[i] and
                  close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Loss of bullish alignment (Lips <= Teeth)
            if lips[i] <= teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Loss of bearish alignment (Teeth <= Jaw)
            if teeth[i] <= jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals