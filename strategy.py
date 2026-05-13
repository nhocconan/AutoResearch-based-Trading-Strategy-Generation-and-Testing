#!/usr/bin/env python3
# 6h_Financial_Force_WeeklyTrend_Volume
# Hypothesis: Combines 6h Elder Ray (Bull/Bear Power) with 1d trend filter and volume spike.
# Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low.
# Long when Bull Power > 0 and rising, Bear Power < 0 and falling, price > 1d EMA50, volume spike.
# Short when Bear Power < 0 and falling, Bull Power > 0 and rising, price < 1d EMA50, volume spike.
# Designed to capture strong directional moves with institutional participation.
# Target: 12-37 trades/year per symbol to minimize fee drag while maintaining edge.

name = "6h_Financial_Force_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')

    # Elder Ray components: Bull Power and Bear Power
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low

    # Trend filter: 1d EMA50
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Volume confirmation: current volume > 2.0 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(13, n):  # Start after sufficient warmup for EMA13
        # Skip if any required value is NaN
        if (np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Bull Power rising, Bear Power falling, price above 1d EMA50, volume spike
            if (bull_power[i] > bull_power[i-1] and 
                bear_power[i] < bear_power[i-1] and 
                close[i] > ema50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power falling, Bull Power rising, price below 1d EMA50, volume spike
            elif (bear_power[i] < bear_power[i-1] and 
                  bull_power[i] > bull_power[i-1] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power turns negative or Bear Power turns positive
            if bull_power[i] < 0 or bear_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power turns positive or Bull Power turns negative
            if bear_power[i] > 0 or bull_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals