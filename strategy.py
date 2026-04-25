#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_1dTrend_ATRStop
Hypothesis: On 4h timeframe, Donchian(20) breakouts with 1d EMA50 trend filter and volume spike (>2.0x 20-bar avg) captures strong directional moves with volume confirmation. Trend filter ensures alignment with daily momentum, reducing false signals in choppy markets. ATR-based stoploss (2.5x ATR) manages risk. Designed for low trade frequency (20-50/year) to minimize fee drag. Works in both bull and bear markets via trend filter.
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
    
    # Get 1d data for HTF trend (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for stoploss
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    high_close[0] = 0
    low_close[0] = 0
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(50, 20, 14)  # EMA50, Donchian20, ATR14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_val = ema_50_aligned[i]
        highest_val = highest_20[i]
        lowest_val = lowest_20[i]
        atr_val = atr[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current volume > 2.0x 20-period average
        volume_spike = vol_val > 2.0 * vol_ma_val
        
        if position == 0:
            # Look for entry signals: Donchian breakout with trend and volume
            # Long: price breaks above upper Donchian with uptrend (close > EMA50) and volume spike
            long_signal = (high_val > highest_val) and (close_val > ema_val) and volume_spike
            # Short: price breaks below lower Donchian with downtrend (close < EMA50) and volume spike
            short_signal = (low_val < lowest_val) and (close_val < ema_val) and volume_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. ATR-based stoploss: price < entry_price - 2.5 * atr
            if close_val < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Opposite Donchian breakout: price breaks below lower Donchian (exit long)
            elif low_val < lowest_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. ATR-based stoploss: price > entry_price + 2.5 * atr
            if close_val > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Opposite Donchian breakout: price breaks above upper Donchian (exit short)
            elif high_val > highest_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_1dTrend_ATRStop"
timeframe = "4h"
leverage = 1.0