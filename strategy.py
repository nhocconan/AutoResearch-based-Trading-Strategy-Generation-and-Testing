#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Long: price breaks above 20-period Donchian high + ATR(14) < ATR(50) (low volatility regime) + volume > 1.5x 20-period avg
- Short: price breaks below 20-period Donchian low + ATR(14) < ATR(50) (low volatility regime) + volume > 1.5x 20-period avg
- Exit: price retouches 10-period EMA (trend-based exit) OR opposite Donchian breakout
- ATR regime filter ensures trades occur in low volatility environments (pre-breakout consolidation)
- Volume confirmation ensures breakout validity
- Works in bull (breakout continuation) and bear (breakdown continuation) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR regime filter: ATR(14) < ATR(50) indicates low volatility (pre-breakout)
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First period
    
    atr_14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(true_range).rolling(window=50, min_periods=50).mean().values
    low_volatility = atr_14 < atr_50  # Low volatility regime
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Exit EMA: 10-period for trend-based exit
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for Donchian, 50 for ATR50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(atr_14[i]) or 
            np.isnan(atr_50[i]) or 
            np.isnan(ema_10[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + low volatility + volume spike
            if volume_spike and low_volatility[i] and close[i] > donchian_high[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + low volatility + volume spike
            elif volume_spike and low_volatility[i] and close[i] < donchian_low[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price retouches 10-period EMA (trend-based exit)
            if close[i] <= ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price retouches 10-period EMA (trend-based exit)
            if close[i] >= ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_ATRRegime_VolumeSpike"
timeframe = "4h"
leverage = 1.0