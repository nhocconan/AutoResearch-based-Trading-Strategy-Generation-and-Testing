#!/usr/bin/env python3
# 6h_keltner_momentum_v1
# Hypothesis: 6h momentum strategy using Keltner Channel breakout with momentum confirmation and volume filter.
# Keltner Channel breakouts capture momentum moves; momentum filter ensures trend strength; volume filter ensures participation.
# Designed to work in both bull and bear markets by capturing strong moves in either direction.
# Target: 20-40 trades/year with ~0.25 position size to minimize fee drag.

name = "6h_keltner_momentum_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Keltner Channel (20, 2.0)
    kc_period = 20
    kc_mult = 2.0
    
    # EMA of typical price for center line
    typical_price = (high + low + close) / 3.0
    kc_center = np.zeros_like(typical_price)
    kc_center[kc_period-1:] = pd.Series(typical_price).ewm(span=kc_period, adjust=False).mean()[kc_period-1:].values
    kc_center[:kc_period-1] = kc_center[kc_period-1]
    
    # ATR for channel width
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.zeros_like(tr)
    atr[kc_period-1:] = pd.Series(tr).ewm(span=kc_period, adjust=False).mean()[kc_period-1:].values
    atr[:kc_period-1] = atr[kc_period-1]
    
    kc_upper = kc_center + kc_mult * atr
    kc_lower = kc_center - kc_mult * atr
    
    # Momentum filter: ROC(10) > 0 for long, < 0 for short
    roc_period = 10
    roc = np.zeros_like(close)
    roc[roc_period:] = (close[roc_period:] - close[:-roc_period]) / close[:-roc_period] * 100
    
    # Volume filter: volume > 1.5x 20-period average
    vol_period = 20
    vol_ma = np.zeros_like(volume)
    vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    vol_ma[:vol_period-1] = vol_ma[vol_period-1]
    
    # Start from sufficient lookback
    start_idx = max(kc_period, roc_period, vol_period) + 5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(roc[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit if momentum fails or price re-enters Keltner Channel
            if roc[i] <= 0 or close[i] <= kc_center[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if momentum fails or price re-enters Keltner Channel
            if roc[i] >= 0 or close[i] >= kc_center[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: break above upper Keltner Channel with positive momentum and volume
            if close[i] > kc_upper[i] and roc[i] > 0 and volume_filter:
                position = 1
                signals[i] = 0.25
            # Short entry: break below lower Keltner Channel with negative momentum and volume
            elif close[i] < kc_lower[i] and roc[i] < 0 and volume_filter:
                position = -1
                signals[i] = -0.25
    
    return signals