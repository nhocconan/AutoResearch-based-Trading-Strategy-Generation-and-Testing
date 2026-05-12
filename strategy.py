#!/usr/bin/env python3
# 6h_12h_1d_MarketStructure_Breakout
# Hypothesis: Combines 6h market structure (HH/HL/LH/LL) with 12h trend filter and volume confirmation.
# Uses higher timeframe structure to avoid false breakouts, targeting 20-40 trades per year.
# Works in bull/bear markets by following 12h trend direction while using 6s structure for precise entries.
# Volume spike confirms institutional interest in breakouts.

name = "6h_12h_1d_MarketStructure_Breakout"
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
    
    # Volume spike: >1.8x 20-period average (on 6h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # 1d data for market structure (swing points)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate swing highs/lows on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Swing high: higher than 2 bars before and after
    swing_high = np.zeros(len(high_1d), dtype=bool)
    swing_low = np.zeros(len(low_1d), dtype=bool)
    for i in range(2, len(high_1d)-2):
        if (high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i-2] and
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            swing_high[i] = True
        if (low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i-2] and
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            swing_low[i] = True
    
    # Last swing high/low levels
    last_swing_high = np.full(len(high_1d), np.nan)
    last_swing_low = np.full(len(low_1d), np.nan)
    
    last_high = np.nan
    last_low = np.nan
    for i in range(len(high_1d)):
        if swing_high[i]:
            last_high = high_1d[i]
        if swing_low[i]:
            last_low = low_1d[i]
        last_swing_high[i] = last_high
        last_swing_low[i] = last_low
    
    # Align swing levels to 6h timeframe
    swing_high_aligned = align_htf_to_ltf(prices, df_1d, last_swing_high)
    swing_low_aligned = align_htf_to_ltf(prices, df_1d, last_swing_low)
    
    # 12h data for trend filter (EMA34)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(swing_high_aligned[i]) or
            np.isnan(swing_low_aligned[i]) or
            np.isnan(ema_34_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above swing high + volume spike + price above 12h EMA34
            if (close[i] > swing_high_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below swing low + volume spike + price below 12h EMA34
            elif (close[i] < swing_low_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below swing high OR closes below 12h EMA34
            if (close[i] < swing_high_aligned[i]) or \
               close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above swing low OR closes above 12h EMA34
            if (close[i] > swing_low_aligned[i]) or \
               close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals