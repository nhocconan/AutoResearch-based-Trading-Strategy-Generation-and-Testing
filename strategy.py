#!/usr/bin/env python3
# 1d_Williams_Alligator_Elder_Ray_Trend
# Hypothesis: Use Williams Alligator (SMMA of median price) to identify trend direction and Elder Ray (bull/bear power) for momentum confirmation.
# Long when price > Alligator teeth (13-period SMMA) and bull power > 0 with rising bear power (bullish momentum).
# Short when price < Alligator teeth and bear power < 0 with rising bull power (bearish momentum).
# Works in trending markets (both bull and bear) by capturing sustained moves with momentum confirmation.
# Uses weekly timeframe for trend filter to avoid whipsaws and ensure alignment with higher timeframe momentum.
# Target: 15-25 trades/year per symbol to minimize fee drag.

name = "1d_Williams_Alligator_Elder_Ray_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(arr, period):
    """Smoothed Moving Average (SMMA) - same as Wilder's smoothing"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    smma_vals = np.full_like(arr, np.nan, dtype=float)
    smma_vals[period-1] = np.mean(arr[:period])
    for i in range(period, len(arr)):
        smma_vals[i] = (smma_vals[i-1] * (period-1) + arr[i]) / period
    return smma_vals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Williams Alligator on weekly timeframe
    median_price_1w = (high_1w + low_1w) / 2
    jaw = smma(median_price_1w, 13)  # Blue line
    teeth = smma(median_price_1w, 8)   # Red line
    lips = smma(median_price_1w, 5)    # Green line
    
    # Align weekly Alligator to daily timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Elder Ray on daily timeframe
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Weekly trend filter: price > teeth for uptrend, price < teeth for downtrend
    weekly_uptrend = close_1w > teeth
    weekly_downtrend = close_1w < teeth
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Warmup for SMMA and EMA
        # Skip if any required value is NaN
        if (np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or 
            np.isnan(weekly_uptrend_aligned[i]) or 
            np.isnan(weekly_downtrend_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above teeth (bullish alignment) AND bull power > 0 AND rising bull power (momentum)
            if (close[i] > teeth_aligned[i] and 
                bull_power[i] > 0 and 
                bull_power[i] > bull_power[i-1] and 
                weekly_uptrend_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below teeth (bearish alignment) AND bear power < 0 AND rising bear power (momentum)
            elif (close[i] < teeth_aligned[i] and 
                  bear_power[i] < 0 and 
                  bear_power[i] < bear_power[i-1] and 
                  weekly_downtrend_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below lips (weaker bullish alignment) OR bull power turns negative
            if close[i] < lips_aligned[i] or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above lips (weaker bearish alignment) OR bear power turns positive
            if close[i] > lips_aligned[i] or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals