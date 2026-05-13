#!/usr/bin/env python3
# 4h_WavTrend_1dTrend_Confirmation
# Hypothesis: WavTrend oscillator (WTO1, WTO2) on 4h timeframe provides early momentum signals.
# Long when WTO1 crosses above WTO2 with 1d uptrend (price > EMA34) and volume confirmation.
# Short when WTO1 crosses below WTO2 with 1d downtrend (price < EMA34) and volume confirmation.
# Exit on opposite cross or trend reversal.
# WavTrend is sensitive to momentum changes, effective in both bull and bear markets.
# Target: 20-40 trades/year to minimize fee drag.

name = "4h_WavTrend_1dTrend_Confirmation"
timeframe = "4h"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate WavTrend on 4h
    # WavTrend parameters: channel_length=10, average_length=21
    hlc3 = (high + low + close) / 3
    esa = pd.Series(hlc3).ewm(span=10, adjust=False, min_periods=10).mean().values
    d = pd.Series(np.abs(hlc3 - esa)).ewm(span=10, adjust=False, min_periods=10).mean().values
    ci = (hlc3 - esa) / (0.015 * d)
    tci = pd.Series(ci).ewm(span=21, adjust=False, min_periods=21).mean().values
    wt1 = tci
    wt2 = pd.Series(wt1).ewm(span=42, adjust=False, min_periods=42).mean().values
    
    # 1d trend: EMA34
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 4h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: volume > 2.0 * 4-period average (2 days worth at 4h)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > 2.0 * vol_ma_4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(wt1[i]) or 
            np.isnan(wt2[i]) or 
            np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        # Detect crossovers
        wt1_cross_above = wt1[i] > wt2[i] and wt1[i-1] <= wt2[i-1]
        wt1_cross_below = wt1[i] < wt2[i] and wt1[i-1] >= wt2[i-1]

        if position == 0:
            # LONG: WTO1 crosses above WTO2 + 1d uptrend + volume spike
            if wt1_cross_above and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: WTO1 crosses below WTO2 + 1d downtrend + volume spike
            elif wt1_cross_below and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: WTO1 crosses below WTO2 or trend reversal
            if wt1_cross_below or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: WTO1 crosses above WTO2 or trend reversal
            if wt1_cross_above or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals