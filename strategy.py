#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + volume confirmation + ATR(14) trailing stop
- Donchian(20) breakouts capture medium-term trends with controlled frequency
- Volume confirmation (> 1.5x 20-period MA) filters false breakouts
- ATR(14) trailing stop (3x ATR) manages risk and reduces whipsaw
- Designed for 4h timeframe targeting 20-50 trades/year to minimize fee drag
- Works in both bull (breakouts catch trends) and bear (short breakdowns) markets
"""

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
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 1.5x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0
    lowest_low_since_entry = 0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20, 14)  # Donchian, volume MA, ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high AND volume confirmation
            if close[i] > donchian_high[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            # Short: price breaks below Donchian low AND volume confirmation
            elif close[i] < donchian_low[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
        else:
            # Update highest/lowest since entry
            if position == 1:
                highest_high_since_entry = max(highest_high_since_entry, high[i])
            else:
                lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            
            # Exit: ATR trailing stop (3x ATR) OR Donchian breakout in opposite direction
            exit_signal = False
            if position == 1:
                # Long exit: price drops 3x ATR from highest high OR breaks below Donchian low
                if close[i] < highest_high_since_entry - 3.0 * atr[i] or close[i] < donchian_low[i]:
                    exit_signal = True
            elif position == -1:
                # Short exit: price rises 3x ATR from lowest low OR breaks above Donchian high
                if close[i] > lowest_low_since_entry + 3.0 * atr[i] or close[i] > donchian_high[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Breakout_VolumeConfirm_ATR_Trail"
timeframe = "4h"
leverage = 1.0