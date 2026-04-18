#!/usr/bin/env python3
"""
6h_Equivolume_SMA_Crossover_VolumeFilter_v1
Hypothesis: On 6h timeframe, price crossing above/below a volume-weighted moving average (VWMA) with volume confirmation captures institutional flow. 
Uses 20-period VWMA as dynamic support/resistance. Volume filter ensures breakouts have institutional participation. 
Works in bull markets via momentum continuation and in bear markets via mean-reversion at extreme deviations from VWMA.
Target: 20-40 trades/year on 6h timeframe (~80-160 total over 4 years).
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
    
    # Calculate 20-period VWMA (Volume Weighted Moving Average) - acts as dynamic support/resistance
    # VWMA = sum(price * volume) / sum(volume) over period
    pv = close * volume
    vwma_numerator = pd.Series(pv).rolling(window=20, min_periods=20).sum().values
    vwma_denominator = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    vwma = np.divide(vwma_numerator, vwma_denominator, 
                     out=np.full_like(vwma_numerator, np.nan), 
                     where=vwma_denominator!=0)
    
    # Volume filter: 1.5x 20-period average volume to filter weak moves
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Price deviation from VWMA for mean-reversion signals in ranging markets
    # When price deviates >2 std dev from VWMA, expect reversion
    vwma_dev = close - vwma
    vwma_std = pd.Series(vwma_dev).rolling(window=20, min_periods=20).std().values
    extreme_dev = np.abs(vwma_dev) > (2.0 * vwma_std)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 40  # Need enough data for VWMA calculation
    
    for i in range(start_idx, n):
        if (np.isnan(vwma[i]) or 
            np.isnan(vol_ma[i]) or
            np.isnan(vwma_std[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vwma_val = vwma[i]
        vol_filt = volume_filter[i]
        ext_dev = extreme_dev[i]
        
        if position == 0:
            # Long signal: price crosses above VWMA with volume (momentum) 
            # OR mean reversion when price is significantly below VWMA
            if price > vwma[i-1] and close[i-1] <= vwma[i-1] and vol_filt:
                signals[i] = 0.25
                position = 1
            elif price < vwma_val and ext_dev and vol_filt:
                # Mean reversion long when price is significantly below VWMA
                signals[i] = 0.25
                position = 1
            # Short signal: price crosses below VWMA with volume (momentum)
            # OR mean reversion when price is significantly above VWMA
            elif price < vwma[i-1] and close[i-1] >= vwma[i-1] and vol_filt:
                signals[i] = -0.25
                position = -1
            elif price > vwma_val and ext_dev and vol_filt:
                # Mean reversion short when price is significantly above VWMA
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position: exit on VWMA cross down or mean reversion signal
            if price < vwma_val or (price > vwma_val and ext_dev and vol_filt):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short position: exit on VWMA cross up or mean reversion signal
            if price > vwma_val or (price < vwma_val and ext_dev and vol_filt):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Equivolume_SMA_Crossover_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0