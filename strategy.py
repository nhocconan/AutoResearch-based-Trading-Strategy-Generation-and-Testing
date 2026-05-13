#!/usr/bin/env python3
# 6h_FisherTransform_Reversal_1dTrend
# Hypothesis: Ehlers Fisher Transform on 6h with 1d trend filter and volume confirmation.
# Fisher Transform identifies extreme price movements and potential reversals.
# Long when Fisher crosses above -1.5 in uptrend with volume spike.
# Short when Fisher crosses below +1.5 in downtrend with volume spike.
# Uses 1d EMA50 for trend filter and volume > 2x 20-period average for confirmation.
# Designed to capture reversals in both bull and bear markets with trend alignment.
# Target: 12-37 trades/year per symbol to minimize fee drag while maintaining edge.

name = "6h_FisherTransform_Reversal_1dTrend"
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

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')

    # Fisher Transform on 6h price
    # Normalize price to [-1, 1] range over lookback period
    def fisher_transform(price_series, length=10):
        # Calculate highest high and lowest low over lookback
        highest_high = pd.Series(price_series).rolling(window=length, min_periods=length).max().values
        lowest_low = pd.Series(price_series).rolling(window=length, min_periods=length).min().values
        
        # Avoid division by zero
        range_hl = highest_high - lowest_low
        range_hl = np.where(range_hl == 0, 1e-10, range_hl)
        
        # Normalize price to [-1, 1]
        value = 2 * ((price_series - lowest_low) / range_hl - 0.5)
        # Clamp to [-0.999, 0.999] to prevent infinity in log
        value = np.clip(value, -0.999, 0.999)
        
        # Fisher Transform
        fish = 0.5 * np.log((1 + value) / (1 - value))
        
        # Smoothed Fisher (signal line)
        fish_smoothed = pd.Series(fish).ewm(span=3, adjust=False, min_periods=3).mean().values
        return fish_smoothed

    fish = fisher_transform(close, length=10)

    # Trend filter: 1d EMA50
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(fish[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Fisher crosses above -1.5 in uptrend with volume spike
            if (fish[i] > -1.5 and fish[i-1] <= -1.5 and 
                close[i] > ema50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Fisher crosses below +1.5 in downtrend with volume spike
            elif (fish[i] < 1.5 and fish[i-1] >= 1.5 and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Fisher crosses below +1.5 or trend turns down
            if fish[i] < 1.5 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Fisher crosses above -1.5 or trend turns up
            if fish[i] > -1.5 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals