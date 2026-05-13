#!/usr/bin/env python3
# 6h_ElderRay_BullPower_1wTrend_Volume
# Hypothesis: Elder Ray Bull Power (bullish pressure) on 6h with 1-week trend filter (EMA50) and volume spike confirmation.
# Bull Power = High - EMA13; Bear Power = Low - EMA13.
# Long when Bull Power > 0 and rising, price above weekly EMA50, and volume spike.
# Short when Bear Power < 0 and falling, price below weekly EMA50, and volume spike.
# Designed to capture momentum in trending markets with institutional-grade filtering.
# Target: 12-37 trades/year per symbol to minimize fee drag while maintaining edge in both bull and bear markets.

name = "6h_ElderRay_BullPower_1wTrend_Volume"
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

    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')

    # Calculate EMA13 for Elder Ray (on 6h data)
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values

    # Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13

    # Slope of Bull/Bear Power (1-period change)
    bull_power_slope = np.diff(bull_power, prepend=bull_power[0])
    bear_power_slope = np.diff(bear_power, prepend=bear_power[0])

    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)

    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(13, n):  # Start after EMA13 warmup
        # Skip if any required value is NaN
        if (np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(bull_power_slope[i]) or 
            np.isnan(bear_power_slope[i]) or 
            np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bull Power positive and rising, price above weekly EMA50, volume spike
            if (bull_power[i] > 0 and 
                bull_power_slope[i] > 0 and 
                close[i] > ema50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power negative and falling, price below weekly EMA50, volume spike
            elif (bear_power[i] < 0 and 
                  bear_power_slope[i] < 0 and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power turns negative or price breaks below weekly EMA50
            if bull_power[i] <= 0 or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power turns positive or price breaks above weekly EMA50
            if bear_power[i] >= 0 or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals