#!/usr/bin/env python3
# 4h_12h_1d_Structure_Reversion
# Hypothesis: Combines 12h trend direction with 1d mean reversion at Bollinger Bands and 4h breakout confirmation.
# Uses 12h EMA50 for trend filter, 1d Bollinger Bands (20,2) for mean-reversion zones, and 4h breakouts for entry timing.
# Volume confirmation (>1.3x 20-period average) filters for institutional participation.
# Designed for low trade frequency (<150 total 4h trades) to minimize fee drag.
# Works in bull/bear markets by following 12h trend while using 1d mean reversion for entries.

name = "4h_12h_1d_Structure_Reversion"
timeframe = "4h"
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
    
    # Volume spike: >1.3x 20-period average (on 4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma)
    
    # 12h EMA for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 1d Bollinger Bands for mean reversion
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    bb_middle = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + (2 * bb_std)
    bb_lower = bb_middle - (2 * bb_std)
    
    # Mean reversion signals: price near Bollinger Bands
    near_bb_lower = close_1d <= bb_lower  # Oversold
    near_bb_upper = close_1d >= bb_upper  # Overbought
    
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    near_bb_lower_aligned = align_htf_to_ltf(prices, df_1d, near_bb_lower.astype(float))
    near_bb_upper_aligned = align_htf_to_ltf(prices, df_1d, near_bb_upper.astype(float))
    
    # 4h swing points for entry timing
    swing_high_4h = np.zeros(len(high), dtype=bool)
    swing_low_4h = np.zeros(len(low), dtype=bool)
    
    for i in range(1, len(high)-1):
        if high[i] > high[i-1] and high[i] > high[i+1]:
            swing_high_4h[i] = True
        if low[i] < low[i-1] and low[i] < low[i+1]:
            swing_low_4h[i] = True
    
    # Calculate 4h swing high and low levels for breakout entries
    last_swing_high_4h = np.full(len(high), np.nan)
    last_swing_low_4h = np.full(len(low), np.nan)
    
    last_high_4h = np.nan
    last_low_4h = np.nan
    
    for i in range(len(high)):
        if swing_high_4h[i]:
            last_high_4h = high[i]
        if swing_low_4h[i]:
            last_low_4h = low[i]
        last_swing_high_4h[i] = last_high_4h
        last_swing_low_4h[i] = last_low_4h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(ema_12h_aligned[i]) or
            np.isnan(bb_lower_aligned[i]) or
            np.isnan(bb_upper_aligned[i]) or
            np.isnan(near_bb_lower_aligned[i]) or
            np.isnan(near_bb_upper_aligned[i]) or
            np.isnan(last_swing_high_4h[i]) or
            np.isnan(last_swing_low_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Uptrend (price > 12h EMA50) + near BB lower + breaks above 4h swing low + volume spike
            if (close[i] > ema_12h_aligned[i] and 
                near_bb_lower_aligned[i] and 
                close[i] > last_swing_low_4h[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend (price < 12h EMA50) + near BB upper + breaks below 4h swing high + volume spike
            elif (close[i] < ema_12h_aligned[i] and 
                  near_bb_upper_aligned[i] and 
                  close[i] < last_swing_high_4h[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 4h swing low OR price crosses above BB middle
            if (close[i] < last_swing_low_4h[i]) or \
               (close[i] > bb_middle[-1] if len(bb_middle) > 0 else False):  # Simplified exit
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above 4h swing high OR price crosses below BB middle
            if (close[i] > last_swing_high_4h[i]) or \
               (close[i] < bb_middle[-1] if len(bb_middle) > 0 else False):  # Simplified exit
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals