#!/usr/bin/env python3
# 4h_price_action_reversion_v1
# Hypothesis: Trade reversals at key price levels (support/resistance) on 4h timeframe with volume confirmation.
# Uses Donchian channel (20) to identify swing highs/lows, volume spike for confirmation, and ATR-based stop.
# Works in both bull and bear markets: in trending markets, pullbacks to channel extremes offer high-probability entries;
# in ranging markets, reversals at channel boundaries capture mean-reversion opportunities.
# Target: 20-40 trades/year on 4h timeframe with strict entry conditions to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_price_action_reversion_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period high/low)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR for volatility and stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 20  # Ensure Donchian and ATR are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.8 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price touches upper Donchian band OR stoploss hit
            if close[i] >= donch_high[i] or close[i] < donch_low[i] + 1.0 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches lower Donchian band OR stoploss hit
            if close[i] <= donch_low[i] or close[i] > donch_high[i] - 1.0 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price touches lower Donchian band with volume surge
            if close[i] <= donch_low[i] and vol_surge:
                position = 1
                signals[i] = 0.25
            # Short entry: price touches upper Donchian band with volume surge
            elif close[i] >= donch_high[i] and vol_surge:
                position = -1
                signals[i] = -0.25
    
    return signals