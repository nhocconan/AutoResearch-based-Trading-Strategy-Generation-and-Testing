#!/usr/bin/env python3
# 4h_donchian_volatility_breakout_v1
# Hypothesis: Donchian channel breakouts with volume confirmation and ATR-based volatility filter.
# Long: price breaks above Donchian(20) high AND volume > 1.5x average AND ATR(14) > 0.02*price
# Short: price breaks below Donchian(20) low AND volume > 1.5x average AND ATR(14) > 0.02*price
# Exit: opposite Donchian breakout or volatility drops below threshold.
# Designed to capture strong trending moves while filtering low-volatility chop and false breakouts.
# Works in both bull and bear markets by trading breakouts in direction of prevailing volatility expansion.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_volatility_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Average volume (20-period)
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # ATR (14-period)
    atr = np.full(n, np.nan)
    tr = np.full(n, np.nan)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr[i] = np.mean(tr[i-14:i])
    
    # Volatility threshold: ATR > 2% of price
    vol_threshold = np.full(n, np.nan)
    for i in range(14, n):
        vol_threshold[i] = 0.02 * close[i]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or \
           np.isnan(avg_volume[i]) or np.isnan(atr[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        # Volatility filter: ATR > threshold
        volatility_ok = atr[i] > vol_threshold[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR volatility drops
            if close[i] < lowest_low[i] or not volatility_ok:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR volatility drops
            if close[i] > highest_high[i] or not volatility_ok:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian high WITH volume and volatility
            if close[i] > highest_high[i] and volume_confirmed and volatility_ok:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian low WITH volume and volatility
            elif close[i] < lowest_low[i] and volume_confirmed and volatility_ok:
                position = -1
                signals[i] = -0.25
    
    return signals