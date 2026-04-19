#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d Donchian breakout, volume confirmation, and ADX trend filter.
# Targets breakouts in trending markets with volume confirmation to avoid false breakouts.
# Designed to work in both bull and bear markets by capturing strong directional moves.
# Expects 10-30 trades per year per symbol, staying within the 50-150 total trades target over 4 years.
name = "12h_1d_Donchian20_Breakout_Volume_ADX"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 10-period Donchian channels on 1d (using prior day to avoid look-ahead)
    # Highest high of last 10 days (excluding current)
    highest_high = pd.Series(high_1d).rolling(window=10, min_periods=10).max().shift(1).values
    lowest_low = pd.Series(low_1d).rolling(window=10, min_periods=10).min().shift(1).values
    
    # Align Donchian levels to 12h timeframe
    donchian_high = align_htf_to_ltf(prices, df_1d, highest_high)
    donchian_low = align_htf_to_ltf(prices, df_1d, lowest_low)
    
    # ADX(14) calculation on 12h data
    tr = np.maximum(high[1:], close[:-1]) - np.minimum(low[1:], close[:-1])
    tr = np.maximum(tr, np.abs(high[1:] - close[:-1]))
    tr = np.maximum(tr, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])
    
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    plus_di = 100 * plus_dm14 / tr14
    minus_di = 100 * minus_dm14 / tr14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20, 14)
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(adx[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        # ADX filter: trending when ADX > 25
        trending = adx[i] > 25
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and trending market
            if price > donchian_high[i] and volume_ok and trending:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and trending market
            elif price < donchian_low[i] and volume_ok and trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns below Donchian high or ADX drops below 20 (range)
            if price < donchian_high[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns above Donchian low or ADX drops below 20 (range)
            if price > donchian_low[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals