#!/usr/bin/env python3
"""
Hypothesis: 4h-based strategy using Donchian channel breakout with volume confirmation and ATR-based volatility filter. 
The strategy enters long when price breaks above the 20-period Donchian upper band with above-average volume,
and enters short when price breaks below the 20-period Donchian lower band with above-average volume.
Positions are exited when price crosses the midline (average of upper and lower bands) or reverses direction.
Designed for 20-30 trades/year to minimize fee drift while capturing trending moves in both bull and bear markets.
"""
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
    
    # Calculate Donchian channels (20-period)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    mid_band = np.full(n, np.nan)
    
    for i in range(20, n):
        upper_band[i] = np.max(high[i-20:i])
        lower_band[i] = np.min(low[i-20:i])
        mid_band[i] = (upper_band[i] + lower_band[i]) / 2
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Calculate ATR (14-period) for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.mean(tr[i-14:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # need Donchian, volume MA, ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(mid_band[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.2 * 20-period average
        vol_confirmed = volume[i] > 1.2 * vol_ma[i]
        
        # Volatility filter: ATR > 50-period average ATR (avoid low volatility chop)
        if i >= 50:
            atr_ma = np.mean(atr[i-50:i])
            vol_filter = atr[i] > 0.5 * atr_ma  # Only trade in sufficient volatility
        else:
            vol_filter = True  # Not enough data for ATR MA, use default
        
        if position == 0:
            # Long entry: price breaks above upper band with volume and volatility confirmation
            if (close[i] > upper_band[i] and 
                vol_confirmed and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower band with volume and volatility confirmation
            elif (close[i] < lower_band[i] and 
                  vol_confirmed and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price crosses below midline or reverses below lower band
            if close[i] < mid_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above midline or reverses above upper band
            if close[i] > mid_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_VolatilityFilter"
timeframe = "4h"
leverage = 1.0