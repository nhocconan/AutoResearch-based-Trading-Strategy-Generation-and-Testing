#!/usr/bin/env python3
"""
4h Donchian Breakout + 1d ATR Regime + Volume Spike
Hypothesis: Donchian(20) breakouts capture momentum in trending markets. 1d ATR regime filter distinguishes trending (ATR rising) from ranging (ATR falling) markets - only trade breakouts in trending regimes. Volume spike confirms institutional participation. Symmetric long/short logic works in bull/bear markets by following the trend. Target 20-40 trades/year to avoid fee drag.
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
    
    # Get 1d data for ATR regime filter (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr_1d = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        tr_1d[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
    
    atr_1d = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        atr_1d[i] = np.mean(tr_1d[i-13:i+1])
    
    # Calculate ATR regime: trending when ATR rising over 5 periods
    atr_ma_5 = np.full(len(close_1d), np.nan)
    for i in range(5, len(close_1d)):
        atr_ma_5[i] = np.mean(atr_1d[i-4:i+1])
    
    atr_regime = align_htf_to_ltf(prices, df_1d, atr_ma_5 > atr_1d)  # True when ATR rising (trending)
    
    # Calculate Donchian channels (20-period) on 4h data
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Calculate 20-period volume MA for volume confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for Donchian, ATR regime, volume MA
    start_idx = max(20, 14+5, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(atr_regime[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        is_trending = atr_regime[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Look for breakout signals
            # Long: price breaks above Donchian high with volume confirmation in trending regime
            long_breakout = (curr_close > donchian_high[i]) and volume_confirm and is_trending
            # Short: price breaks below Donchian low with volume confirmation in trending regime
            short_breakout = (curr_close < donchian_low[i]) and volume_confirm and is_trending
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit conditions: price closes below Donchian low OR 2.0*ATR stop OR trend regime ends
            atr_14 = np.full(n, np.nan)
            for j in range(14, n):
                tr = np.max([high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1])])
                if j == 14:
                    atr_14[j] = tr
                else:
                    atr_14[j] = 0.9 * atr_14[j-1] + 0.1 * tr
            
            atr_val = atr_14[i] if not np.isnan(atr_14[i]) else 0.0
            if (curr_close < donchian_low[i] or 
                curr_close < (highest_since_entry - 2.0 * atr_val) or 
                not is_trending):
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit conditions: price closes above Donchian high OR 2.0*ATR stop OR trend regime ends
            atr_14 = np.full(n, np.nan)
            for j in range(14, n):
                tr = np.max([high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1])])
                if j == 14:
                    atr_14[j] = tr
                else:
                    atr_14[j] = 0.9 * atr_14[j-1] + 0.1 * tr
            
            atr_val = atr_14[i] if not np.isnan(atr_14[i]) else 0.0
            if (curr_close > donchian_high[i] or 
                curr_close > (lowest_since_entry + 2.0 * atr_val) or 
                not is_trending):
                signals[i] = 0.0
                position = 0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_1dATR_Regime_VolumeSpike"
timeframe = "4h"
leverage = 1.0