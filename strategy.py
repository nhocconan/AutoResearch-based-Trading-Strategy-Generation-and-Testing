#!/usr/bin/env python3
"""
12h Donchian(20) breakout with 1-day ATR filter and volume confirmation
Hypothesis: Breakouts of Donchian(20) channels on 12h timeframe, filtered by
1-day ATR-based volatility regime and volume spike, capture strong momentum
with reduced whipsaws. Designed for ~15-25 trades/year to minimize fee drag
in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_atr_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1-day ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volatility regime: ATR > 20-period average (high volatility)
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    vol_regime = atr > (atr_ma * 0.8)  # Active when ATR is not too low
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr[i]) or 
            np.isnan(vol_regime[i]) or 
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR drops too low OR price breaks below Donchian(10) low
            donchian_low = np.min(low[max(0, i-10):i+1])
            if (not vol_regime[i] or 
                close[i] <= donchian_low):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR drops too low OR price breaks above Donchian(10) high
            donchian_high = np.max(high[max(0, i-10):i+1])
            if (not vol_regime[i] or 
                close[i] >= donchian_high):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Donchian(20) channels
            donchian_high = np.max(high[max(0, i-20):i])
            donchian_low = np.min(low[max(0, i-20):i])
            
            # Long: price breaks above Donchian(20) high + vol regime + vol spike
            if (close[i] > donchian_high and
                vol_regime[i] and
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian(20) low + vol regime + vol spike
            elif (close[i] < donchian_low and
                  vol_regime[i] and
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals