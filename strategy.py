#!/usr/bin/env python3
# 12h_Camarilla_Pivot_Trend_Volume
# Hypothesis: Uses daily Camarilla pivot levels (R1, S1) for entry signals and 12h EMA for trend filtering.
# Enters long when price breaks above R1 with 12h EMA uptrend, short when breaks below S1 with 12h EMA downtrend.
# Volume confirmation (2x average) filters false breakouts. ATR-based stop loss manages risk.
# Designed for 12h timeframe with daily pivot levels as structure and trend filter to reduce whipsaw.
# Target: 15-30 trades/year per symbol with disciplined risk management for both bull and bear markets.

name = "12h_Camarilla_Pivot_Trend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
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
    if len(tr) >= 14:
        atr[13] = np.mean(tr[0:14])
        for i in range(14, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Get daily data for Camarilla pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (R1, S1)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1 = pivot + (range_1d * 1.1 / 12)
    s1 = pivot - (range_1d * 1.1 / 12)
    
    # Align pivot levels to 12h timeframe (wait for daily close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Get 12h EMA for trend filter
    ema_period = 20
    ema_12h = np.full_like(close, np.nan)
    if len(close) >= ema_period:
        ema_12h[ema_period-1] = np.mean(close[0:ema_period])
        for i in range(ema_period, len(close)):
            ema_12h[i] = (close[i] * 2 + ema_12h[i-1] * (ema_period - 1)) / (ema_period + 1)
    
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
    
    start_idx = max(20, ema_period-1, 1)  # Need volume MA, EMA, and pivot data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_12h[i]) or np.isnan(volume_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend direction from 12h EMA
        ema_uptrend = close[i] > ema_12h[i]
        
        if position == 0:
            # Enter long: price breaks above R1 + EMA uptrend + volume confirmation
            if close[i] > r1_aligned[i] and ema_uptrend and volume_ratio[i] > 2.0:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 + EMA downtrend + volume confirmation
            elif close[i] < s1_aligned[i] and not ema_uptrend and volume_ratio[i] > 2.0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1 or EMA turns downtrend or ATR stop
            if close[i] < s1_aligned[i] or not ema_uptrend or close[i] < ema_12h[i] - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 or EMA turns uptrend or ATR stop
            if close[i] > r1_aligned[i] or ema_uptrend or close[i] > ema_12h[i] + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals