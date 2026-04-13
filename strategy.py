#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 1d Donchian breakout + volume confirmation + ATR filter
# Strategy: Long when price breaks above 1d Donchian high with volume > 1.5x average
# Short when price breaks below 1d Donchian low with volume > 1.5x average
# Uses Donchian channels from 1d timeframe for strong support/resistance
# Volume surge confirms breakout strength at key levels
# ATR filter ensures volatility is sufficient to avoid false breakouts
# Target: 75-200 total trades over 4 years (19-50/year) for optimal balance

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period Donchian channels on 1d data
    # Using rolling window for high and low
    high_roll = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, high_roll)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, low_roll)
    
    # Volume average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR calculation (14-period) for volatility filter
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume surge condition
        volume_surge = volume[i] > 1.5 * vol_ma_20[i]
        
        # ATR filter: ensure volatility is sufficient
        atr_filter = atr[i] > 0.01 * close[i]  # ATR > 1% of price
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_high_aligned[i]
        short_breakout = close[i] < donchian_low_aligned[i]
        
        # Entry logic
        long_entry = long_breakout and volume_surge and atr_filter
        short_entry = short_breakout and volume_surge and atr_filter
        
        # Exit conditions: opposite breakout or mean reversion to mid-point
        donchian_mid = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
        exit_long = position == 1 and close[i] < donchian_mid
        exit_short = position == -1 and close[i] > donchian_mid
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_breakout_volume_atr_v1"
timeframe = "4h"
leverage = 1.0