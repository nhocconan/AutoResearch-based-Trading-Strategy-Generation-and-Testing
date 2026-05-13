#!/usr/bin/env python3
# 6h_Williams_Alligator_ElderRay_1wTrend
# Hypothesis: Combine Williams Alligator (Jaw/Teeth/Lips) for trend direction with Elder Ray (Bull/Bear Power) for momentum strength, filtered by weekly trend (price above/below weekly EMA50). Enter long when Elder Ray Bull Power > 0 and price above Alligator Teeth in weekly uptrend; short when Bear Power < 0 and price below Alligator Teeth in weekly downtrend. Uses Williams Alligator smoothed with SMMA ( Wilder's smoothing) to avoid whipsaw. Designed for 6B timeframe to capture sustained trends with low trade frequency in both bull and bear markets.

name = "6h_Williams_Alligator_ElderRay_1wTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) aka Wilder's smoothing."""
    if length < 1:
        return source
    result = np.full_like(source, np.nan, dtype=np.float64)
    # First value is simple average
    result[length-1] = np.mean(source[:length])
    # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CURRENT_VALUE) / length
    for i in range(length, len(source)):
        result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values

    # Get weekly data for trend filter and Alligator
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    # Williams Alligator on weekly timeframe (13,8,5 smoothed with SMMA)
    # Jaw (13-period SMMA of median price, shifted 8 bars)
    median_price_1w = (df_1w['high'].values + df_1w['low'].values) / 2
    jaw_raw = smma(median_price_1w, 13)
    jaw = np.roll(jaw_raw, 8)  # shift forward 8 bars
    jaw[0:8] = np.nan  # first 8 values invalid due to shift
    
    # Teeth (8-period SMMA of median price, shifted 5 bars)
    teeth_raw = smma(median_price_1w, 8)
    teeth = np.roll(teeth_raw, 5)  # shift forward 5 bars
    teeth[0:5] = np.nan  # first 5 values invalid due to shift
    
    # Lips (5-period SMMA of median price)
    lips = smma(median_price_1w, 5)

    # Align Alligator lines to 6t timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)

    # Elder Ray on 6t timeframe: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(13, n):  # start after EMA13 warmup
        # Skip if any required value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(jaw_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bull Power > 0 (bullish momentum) + price above Alligator Teeth (uptrend structure) + weekly uptrend
            if (bull_power[i] > 0 and 
                close[i] > teeth_aligned[i] and
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 (bearish momentum) + price below Alligator Teeth (downtrend structure) + weekly downtrend
            elif (bear_power[i] < 0 and 
                  close[i] < teeth_aligned[i] and
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bear Power < 0 (loss of bullish momentum) or price below Alligator Teeth (trend damage)
            if (bear_power[i] < 0 or close[i] < teeth_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bull Power > 0 (loss of bearish momentum) or price above Alligator Teeth (trend damage)
            if (bull_power[i] > 0 or close[i] > teeth_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals