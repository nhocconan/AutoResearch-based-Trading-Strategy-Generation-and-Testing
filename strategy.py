#!/usr/bin/env python3
# 4h_Adaptive_Donchian_Breakout_With_Volume
# Hypothesis: Donchian channel breakouts with volume confirmation and ATR-based position sizing.
# In bull markets: breakout above upper band signals momentum continuation.
# In bear markets: breakdown below lower band signals continuation of downtrend.
# Volume filter ensures institutional participation, reducing false breakouts.
# Uses dynamic position sizing based on ATR volatility to adapt to market conditions.
# Designed for 20-40 trades/year to minimize fee drag.

name = "4h_Adaptive_Donchian_Breakout_With_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Donchian Channel (20-period)
    lookback = 20
    upper = np.full_like(high, np.nan)
    lower = np.full_like(low, np.nan)
    
    for i in range(lookback, n):
        upper[i] = np.max(high[i-lookback:i])
        lower[i] = np.min(low[i-lookback:i])

    # ATR for volatility-based position sizing (14-period)
    atr_period = 14
    tr = np.zeros_like(close)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.zeros_like(close)
    atr[atr_period-1] = np.mean(tr[1:atr_period])
    for i in range(atr_period, n):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(max(lookback, atr_period, 20), n):
        # Skip if data is not ready
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(atr[i]) or np.isnan(volume_spike[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above upper Donchian band with volume spike
            if close[i] > upper[i] and volume_spike[i]:
                # Scale position by volatility (inverse ATR)
                vol_factor = np.clip(atr[0] / atr[i], 0.5, 2.0)  # Normalize to initial volatility
                base_size = 0.25
                signals[i] = base_size * vol_factor
                position = 1
            # SHORT: Price breaks below lower Donchian band with volume spike
            elif close[i] < lower[i] and volume_spike[i]:
                vol_factor = np.clip(atr[0] / atr[i], 0.5, 2.0)
                base_size = 0.25
                signals[i] = -base_size * vol_factor
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters Donchian channel (below midpoint)
            midpoint = (upper[i] + lower[i]) / 2
            if close[i] < midpoint:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position with volatility scaling
                vol_factor = np.clip(atr[0] / atr[i], 0.5, 2.0)
                base_size = 0.25
                signals[i] = base_size * vol_factor
        elif position == -1:
            # EXIT SHORT: Price re-enters Donchian channel (above midpoint)
            midpoint = (upper[i] + lower[i]) / 2
            if close[i] > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                vol_factor = np.clip(atr[0] / atr[i], 0.5, 2.0)
                base_size = 0.25
                signals[i] = -base_size * vol_factor

    return signals