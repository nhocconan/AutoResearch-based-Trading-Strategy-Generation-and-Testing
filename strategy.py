#!/usr/bin/env python3
# 1h_SuperTrend_Filter_4hTrend_Volume
# Hypothesis: Use 4h SuperTrend for trend direction and 1h SuperTrend for entry timing with volume confirmation.
# This combines trend following with volatility-based dynamic support/resistance to capture trends
# while avoiding whipsaws in ranging markets. Works in both bull and bear by following the trend.
# Target: 15-37 trades/year per symbol to minimize fee drag.

name = "1h_SuperTrend_Filter_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def supertrend(high, low, close, period=10, multiplier=3):
    """Calculate SuperTrend indicator."""
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    # Calculate ATR
    atr = np.zeros_like(close)
    atr[:period] = np.nan
    if len(close) > period:
        atr[period] = np.nanmean(tr[1:period+1])
        for i in range(period+1, len(close)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    # Calculate basic upper and lower bands
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize SuperTrend
    supertrend = np.full_like(close, np.nan)
    direction = np.full_like(close, 1)  # 1 for uptrend, -1 for downtrend
    
    # Set first valid value
    if len(close) > period:
        supertrend[period] = upper_band[period]
        direction[period] = 1
    
    for i in range(period+1, len(close)):
        # Calculate bands
        upper_band[i] = hl2[i] + multiplier * atr[i]
        lower_band[i] = hl2[i] - multiplier * atr[i]
        
        # Apply SuperTrend logic
        if close[i-1] > supertrend[i-1]:
            # Previous close was above previous SuperTrend (uptrend)
            supertrend[i] = max(lower_band[i], supertrend[i-1])
            if supertrend[i] < supertrend[i-1]:
                supertrend[i] = supertrend[i-1]
            direction[i] = 1
        else:
            # Previous close was below previous SuperTrend (downtrend)
            supertrend[i] = min(upper_band[i], supertrend[i-1])
            if supertrend[i] > supertrend[i-1]:
                supertrend[i] = supertrend[i-1]
            direction[i] = -1
    
    return supertrend, direction, atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h SuperTrend for trend direction
    st_4h, dir_4h, atr_4h = supertrend(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, period=10, multiplier=3)
    dir_4h_aligned = align_htf_to_ltf(prices, df_4h, dir_4h)

    # Calculate 1h SuperTrend for entry timing
    st_1h, dir_1h, atr_1h = supertrend(high, low, close, period=10, multiplier=3)

    # Volume confirmation: current volume > 1.5 x 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.nan
        else:
            vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after sufficient warmup
        # Skip if any required value is NaN
        if (np.isnan(dir_4h_aligned[i]) or 
            np.isnan(st_1h[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: 4h uptrend AND price above 1h SuperTrend with volume spike
            if (dir_4h_aligned[i] == 1 and 
                close[i] > st_1h[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: 4h downtrend AND price below 1h SuperTrend with volume spike
            elif (dir_4h_aligned[i] == -1 and 
                  close[i] < st_1h[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: 4h trend turns down OR price falls below 1h SuperTrend
            if dir_4h_aligned[i] == -1 or close[i] < st_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: 4h trend turns up OR price rises above 1h SuperTrend
            if dir_4h_aligned[i] == 1 or close[i] > st_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20

    return signals