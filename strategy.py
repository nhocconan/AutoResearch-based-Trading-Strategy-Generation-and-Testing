#!/usr/bin/env python3
# 4h_12h_DonchianBreakout_VolumeFilter_Trend
# Hypothesis: Uses 4h Donchian channel breakouts filtered by 12h EMA trend and volume spike.
# Goes long on upper band breakout when 12h EMA is rising and volume > 2x average.
# Goes short on lower band breakout when 12h EMA is falling and volume > 2x average.
# Includes ATR-based stop loss to limit drawdowns. Designed for 4h timeframe with 12h trend filter
# to reduce whipsaw in both bull and bear markets. Target: 20-40 trades/year.

name = "4h_12h_DonchianBreakout_VolumeFilter_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.full_like(tr, np.nan)
    if len(tr) >= 10:
        atr[9] = np.mean(tr[0:10])
        for i in range(10, len(tr)):
            atr[i] = (atr[i-1] * 9 + tr[i]) / 10
    
    # Donchian channel (20-period) on 4h
    highest = np.full_like(high, np.nan)
    lowest = np.full_like(low, np.nan)
    for i in range(19, len(high)):
        highest[i] = np.max(high[i-19:i+1])
        lowest[i] = np.min(low[i-19:i+1])
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA(34) for trend
    ema_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 34:
        ema_12h[33] = np.mean(close_12h[0:34])
        for i in range(34, len(close_12h)):
            ema_12h[i] = (close_12h[i] * 0.0566) + (ema_12h[i-1] * 0.9434)  # 2/(34+1)
    
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume filter: current volume vs 20-period average
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Need EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest[i]) or np.isnan(lowest[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(volume_ratio[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 12h EMA trend (rising/falling)
        if i >= 1:
            ema_rising = ema_12h_aligned[i] > ema_12h_aligned[i-1]
            ema_falling = ema_12h_aligned[i] < ema_12h_aligned[i-1]
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Enter long: price breaks above upper Donchian band + 12h EMA rising + volume spike
            if close[i] > highest[i] and ema_rising and volume_ratio[i] > 2.0:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian band + 12h EMA falling + volume spike
            elif close[i] < lowest[i] and ema_falling and volume_ratio[i] > 2.0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price closes below lower Donchian band or ATR stop
            if close[i] < lowest[i] or close[i] < (highest[i] - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above upper Donchian band or ATR stop
            if close[i] > highest[i] or close[i] > (lowest[i] + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals