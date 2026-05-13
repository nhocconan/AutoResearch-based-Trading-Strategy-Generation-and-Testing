#!/usr/bin/env python3
# 6h_ElderRay_ZeroCross_1dTrend_Volume
# Hypothesis: Elder Ray (Bull/Bear Power) crosses zero when bullish/bearish momentum shifts.
# Enter long when Bull Power crosses above zero with 1d uptrend (EMA50) and volume confirmation.
# Enter short when Bear Power crosses below zero with 1d downtrend and volume confirmation.
# Exit on opposite cross or trend reversal. Works in bull/bear via trend filter.
# Target: 15-35 trades/year per symbol.

name = "6h_ElderRay_ZeroCross_1dTrend_Volume"
timeframe = "6h"
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

    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50 trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 13-period EMA on 6h)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Detect zero crosses
    bull_cross_up = (bull_power[1:] > 0) & (bull_power[:-1] <= 0)
    bear_cross_down = (bear_power[1:] < 0) & (bear_power[:-1] >= 0)
    # Prepend False for index 0
    bull_cross_up = np.concatenate([[False], bull_cross_up])
    bear_cross_down = np.concatenate([[False], bear_cross_down])
    
    # Volume spike: volume > 2.0 * 20-period average (~6.7 hours)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(13, n):  # start after EMA13 warmup
        # Skip if any required value is NaN
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bull Power crosses above zero + 1d uptrend + volume spike
            if bull_cross_up[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power crosses below zero + 1d downtrend + volume spike
            elif bear_cross_down[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bear Power crosses below zero or trend reversal
            if bear_cross_down[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bull Power crosses above zero or trend reversal
            if bull_cross_up[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals