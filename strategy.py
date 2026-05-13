#!/usr/bin/env python3
# 4h_ADX_Trend_With_Volume_Spike
# Hypothesis: ADX > 25 identifies strong trends, and volume spikes confirm institutional participation.
# Enter long when ADX > 25, +DI crosses above -DI, and volume > 1.5x 20-period average.
# Enter short when ADX > 25, -DI crosses above +DI, and volume > 1.5x 20-period average.
# Exit when ADX < 20 (trend weakens) or opposite DI crossover occurs.
# Uses 4h timeframe with volume confirmation to filter false breakouts.
# Designed to work in both bull (trend following long) and bear (trend following short).
# Target: 20-40 trades/year per symbol.

name = "4h_ADX_Trend_With_Volume_Spike"
timeframe = "4h"
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

    # Calculate ADX and DI components
    # True Range
    tr0 = np.abs(high - low)
    tr1 = np.abs(high - np.roll(close, 1))
    tr2 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr0, np.maximum(tr1, tr2))
    tr[0] = tr0[0]

    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Smooth TR, +DM, -DM over 14 periods (Wilder's smoothing)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value: simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result

    atr = wilders_smooth(tr, 14)
    plus_dm_smooth = wilders_smooth(plus_dm, 14)
    minus_dm_smooth = wilders_smooth(minus_dm, 14)

    # Calculate +DI and -DI
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    for i in range(14, n):
        if atr[i] != 0:
            plus_di[i] = (plus_dm_smooth[i] / atr[i]) * 100
            minus_di[i] = (minus_dm_smooth[i] / atr[i]) * 100

    # Calculate DX and ADX
    dx = np.full(n, np.nan)
    for i in range(14, n):
        if (plus_di[i] + minus_di[i]) != 0:
            dx[i] = (np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100

    adx = wilders_smooth(dx, 14)

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(28, n):  # Start after ADX is ready
        # Skip if data is not ready
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: ADX > 25, +DI crosses above -DI, volume spike
            if (adx[i] > 25 and 
                plus_di[i] > minus_di[i] and 
                plus_di[i-1] <= minus_di[i-1] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: ADX > 25, -DI crosses above +DI, volume spike
            elif (adx[i] > 25 and 
                  minus_di[i] > plus_di[i] and 
                  minus_di[i-1] <= plus_di[i-1] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: ADX < 20 (trend weak) OR -DI crosses above +DI
            if (adx[i] < 20) or (minus_di[i] > plus_di[i] and minus_di[i-1] <= plus_di[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: ADX < 20 (trend weak) OR +DI crosses above -DI
            if (adx[i] < 20) or (plus_di[i] > minus_di[i] and plus_di[i-1] <= minus_di[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals