#!/usr/bin/env python3
name = "6h_ADX_Williams_Alligator_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # ADX filter (14-period) - trend strength
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    up_move = np.diff(high, prepend=high[0])
    down_move = np.diff(np.roll(low, 1), prepend=low[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / (atr14 * 14 + 1e-10)
    minus_di14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / (atr14 * 14 + 1e-10)
    dx = 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14 + 1e-10)
    adx14 = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Williams Alligator (13,8,5 SMAs shifted 8,5,3)
    def sma(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).mean().values
    
    jaw = sma(close, 13)  # Teeth
    teeth = sma(close, 8)  # Teeth
    lips = sma(close, 5)   # Lips
    
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Williams Alligator signals: Lips above Teeth above Jaw = uptrend
    # Lips below Teeth below Jaw = downtrend
    alligator_up = (lips_shifted > teeth_shifted) & (teeth_shifted > jaw_shifted)
    alligator_down = (lips_shifted < teeth_shifted) & (teeth_shifted < jaw_shifted)
    
    # Williams %R (14-period) - momentum oscillator
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Williams %R overbought/oversold
    williams_oversold = williams_r < -80
    williams_overbought = williams_r > -20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 13)  # Wait for Williams %R and Alligator
    
    for i in range(start_idx, n):
        if np.isnan(adx14[i]) or np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or np.isnan(williams_r[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: ADX > 25 (strong trend) + Alligator aligned up + Williams %R oversold
            if adx14[i] > 25 and alligator_up[i] and williams_oversold[i]:
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25 + Alligator aligned down + Williams %R overbought
            elif adx14[i] > 25 and alligator_down[i] and williams_overbought[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: ADX < 20 (weak trend) or Alligator alignment breaks
            if adx14[i] < 20 or not alligator_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: ADX < 20 or Alligator alignment breaks
            if adx14[i] < 20 or not alligator_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Combines ADX trend strength with Williams Alligator for trend direction and Williams %R for timing entries.
# Long when: ADX > 25 (strong trend) + Alligator aligned bullish (Lips > Teeth > Jaw) + Williams %R oversold (< -80)
# Short when: ADX > 25 + Alligator aligned bearish (Lips < Teeth < Jaw) + Williams %R overbought (> -20)
# Exits when trend weakens (ADX < 20) or Alligator alignment breaks.
# Williams %R provides entry timing within strong trends, avoiding chop.
# Works in both bull and bear markets by capturing strong trends with proper timing.