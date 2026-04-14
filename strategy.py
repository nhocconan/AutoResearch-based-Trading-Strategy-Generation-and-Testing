#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = volumes['volume'].values
    
    # Load 12h data once for higher timeframe context
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h ATR for volatility filter
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = abs(df_12h['high'] - df_12h['close'].shift(1))
    tr3 = abs(df_12h['low'] - df_12h['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr.rolling(window=14, min_periods=14).mean().values
    
    # 12h median volume for volume spike filter
    vol_median_12h = np.nanmedian(df_12h['volume'].values)
    
    # 4h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(close).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate median volume for 4h volume spike
    vol_median_4h = np.nanmedian(volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(100, n):
        # Get aligned 12h ATR and median volume
        atr_12h_i = align_htf_to_ltf(prices, df_12h, atr_12h)[i]
        vol_median_12h_i = align_htf_to_ltf(prices, df_12h, np.full_like(atr_12h, vol_median_12h))[i]
        
        if np.isnan(atr_12h_i) or np.isnan(vol_median_12h_i):
            continue
        
        # Volatility filter: require sufficient volatility
        vol_filter = atr_12h_i > (np.nanmean(atr_12h) * 0.8)
        
        # Volume spike filters
        vol_spike_4h = volume[i] > 1.5 * vol_median_4h
        vol_spike_12h = volume[i] > 1.5 * vol_median_12h_i
        
        # Long conditions:
        # 1. Price breaks above Donchian high (breakout)
        # 2. 12h volatility filter
        # 3. Volume spike on either timeframe
        if position == 0 and vol_filter and (vol_spike_4h or vol_spike_12h):
            if close[i] > highest_high[i]:
                position = 1
                signals[i] = position_size
            # Short conditions:
            # 1. Price breaks below Donchian low (breakdown)
            elif close[i] < lowest_low[i]:
                position = -1
                signals[i] = -position_size
        
        # Exit conditions: price crosses back to middle of channel
        elif position == 1:
            if close[i] < (highest_high[i] + lowest_low[i]) / 2:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            if close[i] > (highest_high[i] + lowest_low[i]) / 2:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_12h_Donchian_Vol_Volatility_Filter"
timeframe = "4h"
leverage = 1.0