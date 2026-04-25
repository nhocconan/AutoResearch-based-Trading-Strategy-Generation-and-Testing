#!/usr/bin/env python3
"""
4h Donchian(20) Breakout + 1d ATR Filter + Volume Spike + ATR Trailing Stop
Hypothesis: Donchian channel breakouts capture strong momentum moves. The 1d ATR filter ensures we only trade when volatility is elevated (avoiding choppy low-vol periods). Volume spike confirms breakout strength. ATR-based trailing stop limits downside and lets winners run. Works in bull/bear markets by trading breakouts in direction of elevated volatility regime.
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
    
    # Get 1d data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate 4h ATR(14) for stoploss and position sizing
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for Donchian and ATR warmup
    start_idx = lookback
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(atr_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        atr_val = atr[i]
        atr_1d_val = atr_1d_aligned[i]
        
        # Volume spike: current volume > 1.5 * 20-period average (less strict than 2.0 to avoid too few trades)
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 1.5 * vol_ma_20
        
        # Volatility regime: 1d ATR > its 50-period average (avoid low-vol chop)
        if i >= 50:
            atr_1d_ma_50 = np.mean(atr_1d_aligned[i-49:i+1])
        else:
            atr_1d_ma_50 = np.mean(atr_1d_aligned[:i+1])
        high_vol_regime = atr_1d_val > atr_1d_ma_50
        
        if position == 0:
            # Long: price breaks above Donchian upper band AND high vol regime AND volume spike
            long_condition = (curr_close > highest_high[i]) and high_vol_regime and volume_spike
            # Short: price breaks below Donchian lower band AND high vol regime AND volume spike
            short_condition = (curr_close < lowest_low[i]) and high_vol_regime and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_high
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_low
        elif position == 1:
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # Check trailing stop: 2.5 * ATR below highest since entry
            if curr_close <= highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
            # Exit long: price returns below Donchian middle (optional re-entry prevention)
            elif curr_close < (highest_high[i] + lowest_low[i]) / 2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Check trailing stop: 2.5 * ATR above lowest since entry
            if curr_close >= lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
            # Exit short: price returns above Donchian middle
            elif curr_close > (highest_high[i] + lowest_low[i]) / 2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dATRFilter_VolumeSpike_ATRTrail_v1"
timeframe = "4h"
leverage = 1.0