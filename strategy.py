#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and ADX trend filter
# Uses daily Donchian breakout for trend direction, confirmed by volume spike and ADX>25.
# Designed to work in both bull and bear markets by requiring strong trend + volume.
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day Donchian channels for trend direction
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period Donchian channels
    high_max = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 12h timeframe
    donchian_high = align_htf_to_ltf(prices, df_1d, high_max)
    donchian_low = align_htf_to_ltf(prices, df_1d, low_min)
    
    # 1-day ADX for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    tr = np.maximum(high_1d[1:] - low_1d[1:], np.absolute(high_1d[1:] - close_1d[:-1]), np.absolute(low_1d[1:] - close_1d[:-1]))
    
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 1-day volume confirmation
    vol_1d = df_1d['volume'].values
    vol_ma = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_threshold = 2.0 * vol_ma
    vol_aligned = align_htf_to_ltf(prices, df_1d, vol_threshold)
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_aligned[i])):
            continue
        
        # Long: price breaks above Donchian high, ADX > 25, volume spike
        if (close[i] > donchian_high[i] and 
            adx_aligned[i] > 25 and 
            volume[i] > vol_aligned[i]):
            signals[i] = 0.25
        
        # Short: price breaks below Donchian low, ADX > 25, volume spike
        elif (close[i] < donchian_low[i] and 
              adx_aligned[i] > 25 and 
              volume[i] > vol_aligned[i]):
            signals[i] = -0.25
        
        # Exit: price returns to middle of channel or ADX weakens
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (close[i] < (donchian_high[i] + donchian_low[i]) / 2 or adx_aligned[i] <= 25)) or
               (signals[i-1] == -0.25 and (close[i] > (donchian_high[i] + donchian_low[i]) / 2 or adx_aligned[i] <= 25)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_Donchian_Breakout_Volume_ADX"
timeframe = "12h"
leverage = 1.0