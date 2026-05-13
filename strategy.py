#!/usr/bin/env python3
# 6h_ElderRay_BullBearPower_1dTrend_Volume
# Hypothesis: Elder Ray's Bull Power (high - EMA13) and Bear Power (EMA13 - low) measure bull/bear strength relative to trend.
# Go long when Bull Power > 0 (bulls in control) with 1d uptrend and volume confirmation.
# Go short when Bear Power > 0 (bears in control) with 1d downtrend and volume confirmation.
# Works in bull markets (Bull Power > 0 in uptrend) and bear markets (Bear Power > 0 in downtrend).
# Target: 15-35 trades/year per symbol to minimize fee drag.

name = "6h_ElderRay_BullBearPower_1dTrend_Volume"
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

    # Get 1d data for EMA13 and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA13 on 1d close
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components on 1d
    bull_power_1d = df_1d['high'].values - ema13_1d  # Bull Power = High - EMA13
    bear_power_1d = ema13_1d - df_1d['low'].values   # Bear Power = EMA13 - Low
    
    # 1d trend: EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike: volume > 2.0 * 3-period average (1.5 days worth at 6h)
    vol_ma_3 = pd.Series(volume).rolling(window=3, min_periods=3).mean().values
    volume_spike = volume > 2.0 * vol_ma_3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(100, n):
        # Skip if any required value is NaN
        if (np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bull Power > 0 (bulls in control) + 1d uptrend + volume spike
            if bull_power_aligned[i] > 0 and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power > 0 (bears in control) + 1d downtrend + volume spike
            elif bear_power_aligned[i] > 0 and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power <= 0 or trend reversal
            if bull_power_aligned[i] <= 0 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power <= 0 or trend reversal
            if bear_power_aligned[i] <= 0 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals