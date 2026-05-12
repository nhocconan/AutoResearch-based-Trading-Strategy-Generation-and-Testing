#!/usr/bin/env python3
# 12h_ThreeBar_Reversal_Volume
# Hypothesis: On 12h timeframe, look for 3-bar reversal patterns (bullish: higher low, higher close; bearish: lower high, lower close) with volume confirmation and 1-day ADX trend filter.
# Enter long on bullish 3-bar reversal when volume > 1.5x 20-period average and ADX > 25 (trending market).
# Enter short on bearish 3-bar reversal under same conditions.
# Exit when opposite 3-bar reversal forms or volume drops below average.
# Designed to capture momentum shifts in both bull and bear markets with low trade frequency.

name = "12h_ThreeBar_Reversal_Volume"
timeframe = "12h"
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
    
    # Load daily data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate ADX (14-period)
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False).mean() / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean()
    
    adx_values = adx.values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(adx_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        vol_confirm = volume_confirm[i]
        
        # Check for 3-bar reversal patterns
        if i >= 2:
            # Bullish 3-bar reversal: higher low, higher close
            bullish_reversal = (low[i] > low[i-1]) and (low[i-1] > low[i-2]) and \
                              (close[i] > close[i-1]) and (close[i-1] > close[i-2])
            # Bearish 3-bar reversal: lower high, lower close
            bearish_reversal = (high[i] < high[i-1]) and (high[i-1] < high[i-2]) and \
                              (close[i] < close[i-1]) and (close[i-1] < close[i-2])
        else:
            bullish_reversal = False
            bearish_reversal = False
        
        if position == 0:
            # LONG: Bullish 3-bar reversal with volume confirmation and ADX > 25
            if bullish_reversal and vol_confirm and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish 3-bar reversal with volume confirmation and ADX > 25
            elif bearish_reversal and vol_confirm and adx_val > 25:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish 3-bar reversal or volume drops below average
            if bearish_reversal or not vol_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish 3-bar reversal or volume drops below average
            if bullish_reversal or not vol_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals