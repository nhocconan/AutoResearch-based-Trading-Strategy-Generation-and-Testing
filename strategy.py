#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h 1-day Donchian(20) breakout + weekly volume spike + ADX(14) trend filter.
# Uses higher timeframe structure for direction and lower timeframe for entry timing.
# Long when price breaks above 1-day Donchian high with weekly volume > 2x average and ADX > 25.
# Short when price breaks below 1-day Donchian low with weekly volume > 2x average and ADX > 25.
# Exit when price returns to Donchian midpoint or ADX < 20.
# Designed for low trade frequency (<30/year) to avoid fee drag in choppy markets.
name = "12h_1d_Donchian20_Volume_ADX"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period)
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Align to 12h timeframe
    donch_high_12h = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_12h = align_htf_to_ltf(prices, df_1d, donch_low)
    donch_mid_12h = align_htf_to_ltf(prices, df_1d, donch_mid)
    
    # Get weekly data for volume filter
    df_1w = get_htf_data(prices, '1w')
    volume_1w = df_1w['volume'].values
    volume_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_spike_1w = volume_1w > (volume_ma_1w * 2.0)
    volume_spike_12h = align_htf_to_ltf(prices, df_1w, volume_spike_1w.astype(float))
    
    # Calculate ADX(14) on 12h data
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]), np.absolute(low[1:] - close[:-1]))
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.absolute(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_12h[i]) or np.isnan(donch_low_12h[i]) or 
            np.isnan(donch_mid_12h[i]) or np.isnan(volume_spike_12h[i]) or
            np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + volume spike + ADX > 25
            if (close[i] > donch_high_12h[i] and 
                volume_spike_12h[i] > 0.5 and 
                adx[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume spike + ADX > 25
            elif (close[i] < donch_low_12h[i] and 
                  volume_spike_12h[i] > 0.5 and 
                  adx[i] > 25):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price returns to midpoint OR ADX < 20
            if (close[i] < donch_mid_12h[i]) or (adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price returns to midpoint OR ADX < 20
            if (close[i] > donch_mid_12h[i]) or (adx[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals