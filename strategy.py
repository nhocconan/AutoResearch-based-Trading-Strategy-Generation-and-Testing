#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and ATR filter
# In trending markets, price breaks Donchian channels with volume (trend continuation)
# In ranging markets, price reverts from Donchian extremes with volume (mean reversion)
# Uses volume to distinguish breakouts from fakeouts and ATR to filter low volatility
# Works in both bull and bear markets by adapting to volatility regime
# Target: 25-40 trades/year per symbol (~100-160 total over 4 years)

name = "4h_Donchian20_VolumeATR_Filter"
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
    
    # Calculate Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for volatility filter
    tr1 = np.maximum(high[1:], close[:-1]) - np.minimum(low[1:], close[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need Donchian and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr[i]
        
        # Volume and volatility filters
        volume_confirmed = vol > 1.8 * vol_ma
        volatility_filter = atr_val > np.percentile(atr[max(0, i-100):i+1], 30) if i >= 30 else atr_val > 0
        
        # Donchian levels
        upper = high_max[i]
        lower = low_min[i]
        
        if position == 0:
            # Long conditions:
            # 1. Breakout above upper band with volume and volatility
            # 2. Mean reversion from lower band with volume (oversold bounce)
            if ((price > upper and volume_confirmed and volatility_filter) or 
                (price < lower * 1.02 and price > lower and volume_confirmed and volatility_filter)):
                signals[i] = 0.25
                position = 1
            # Short conditions:
            # 1. Breakdown below lower band with volume and volatility
            # 2. Mean reversion from upper band with volume (overbought rejection)
            elif ((price < lower and volume_confirmed and volatility_filter) or 
                  (price > upper * 0.98 and price < upper and volume_confirmed and volatility_filter)):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: breakdown below lower band or reversal at upper band
            if price < lower or (price > upper * 0.95 and price < upper):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: breakout above upper band or reversal at lower band
            if price > upper or (price < lower * 1.05 and price > lower):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals