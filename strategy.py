#!/usr/bin/env python3
# 6h_ElderRay_BullBearPower_12hTrend_Volume
# Hypothesis: Elder Ray (Bull Power/Bear Power) measures bull/bear strength relative to EMA.
# Combined with 12h EMA trend filter and volume spikes, it captures strong momentum moves.
# Bull Power > 0 and rising indicates bullish strength; Bear Power < 0 and falling indicates bearish strength.
# Volume confirms institutional participation. Designed for 15-30 trades/year on 6h to minimize fee drag.
# Works in bull/bear: long when Bull Power > 0 and rising with volume and above 12h EMA;
# short when Bear Power < 0 and falling with volume and below 12h EMA.

name = "6h_ElderRay_BullBearPower_12hTrend_Volume"
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

    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA34 trend filter
    ema34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)

    # Calculate EMA13 for Elder Ray (using high/low prices)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    ema13_high = high_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_low = low_series.ewm(span=13, adjust=False, min_periods=13).mean().values

    # Elder Ray components
    bull_power = high - ema13_high  # Bull Power = High - EMA13
    bear_power = low - ema13_low    # Bear Power = Low - EMA13

    # Slope of Bull Power and Bear Power (3-period change)
    bull_power_slope = np.diff(bull_power, prepend=bull_power[0])
    bear_power_slope = np.diff(bear_power, prepend=bear_power[0])

    # Volume confirmation: current volume > 1.8 x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(bull_power_slope[i]) or 
            np.isnan(bear_power_slope[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bull Power > 0 and rising, with volume spike and above 12h EMA
            if (bull_power[i] > 0 and 
                bull_power_slope[i] > 0 and 
                volume_spike[i] and 
                close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 and falling, with volume spike and below 12h EMA
            elif (bear_power[i] < 0 and 
                  bear_power_slope[i] < 0 and 
                  volume_spike[i] and 
                  close[i] < ema34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power <= 0 or stops rising or below 12h EMA
            if (bull_power[i] <= 0 or 
                bull_power_slope[i] <= 0 or 
                close[i] < ema34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power >= 0 or stops falling or above 12h EMA
            if (bear_power[i] >= 0 or 
                bear_power_slope[i] >= 0 or 
                close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals