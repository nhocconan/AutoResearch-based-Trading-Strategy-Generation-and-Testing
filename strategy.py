#!/usr/bin/env python3
"""
4h Donchian Breakout + Volume Spike + ATR Regime Filter
Hypothesis: Donchian(20) breakouts capture strong momentum moves. Volume confirmation 
ensures breakout validity, while ATR-based regime filter (low volatility = range, 
high volatility = trend) adapts to market conditions. Works in bull markets (trend 
continuation up) and bear markets (trend continuation down) by using Donchian 
direction as trend filter. Designed for BTC/ETH with 75-200 total trades over 4 years.
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
    
    # Calculate Donchian channels (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # Calculate ATR(14) for volatility regime and stoploss
    atr = np.full(n, np.nan)
    tr = np.full(n, np.nan)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[1:15])
        else:
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate 20-period volume MA for volume spike confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian, ATR, volume MA
    start_idx = max(20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        atr_val = atr[i]
        vol_ma = vol_ma_20[i]
        
        # ATR-based regime filter: volatility expansion = trend regime
        # Use ratio of current ATR to 50-period ATR mean to detect volatility expansion
        if i >= 50:
            atr_ma_50 = np.mean(atr[i-49:i+1])
            vol_ratio = atr_val / atr_ma_50 if atr_ma_50 > 0 else 1.0
            # High volatility regime (trending): vol_ratio > 1.2
            # Low volatility regime (range): vol_ratio <= 1.2
            vol_regime_trending = vol_ratio > 1.2
        else:
            vol_regime_trending = True  # Default to trending for early bars
        
        if position == 0:
            # Look for breakout signals with volume confirmation and volatility regime
            # Long: price breaks above upper channel with volume confirmation in high vol regime
            long_breakout = (curr_close > upper_channel) and (curr_volume > 2.0 * vol_ma) and vol_regime_trending
            # Short: price breaks below lower channel with volume confirmation in high vol regime
            short_breakout = (curr_close < lower_channel) and (curr_volume > 2.0 * vol_ma) and vol_regime_trending
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
                position = 0
        elif position == 1:
            # Exit long: price closes below lower channel OR volatility contraction (range regime)
            if curr_close < lower_channel or not vol_regime_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above upper channel OR volatility contraction (range regime)
            if curr_close > upper_channel or not vol_regime_trending:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_VolumeSpike_ATRRegime"
timeframe = "4h"
leverage = 1.0