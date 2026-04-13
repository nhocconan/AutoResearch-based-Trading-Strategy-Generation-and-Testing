#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1-day volume spike and 1-day Choppiness Index filter
# Long when price breaks above Donchian(20) high AND 1d volume > 1.5x 20-day avg AND CHOP(14) < 38.2 (trending)
# Short when price breaks below Donchian(20) low AND 1d volume > 1.5x 20-day avg AND CHOP(14) < 38.2
# Exit when price crosses opposite Donchian(10) level (faster exit) OR volume drops below average OR CHOP > 61.8 (range)
# Uses daily timeframe for volume and regime filter to reduce whipsaws in ranging markets
# Target: 75-200 total trades over 4 years (19-50/year) with focus on BTC/ETH performance

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for volume and choppiness index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day average volume
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate True Range for Choppiness Index
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], 
                        np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate ATR(14) for denominator
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Calculate Choppiness Index: CHOP = 100 * log10(sum(ATR14) / (HH14 - LL14)) / log10(14)
    # Using sum of ATR over 14 periods
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    range_14 = hh_14 - ll_14
    # Avoid division by zero
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    chop = 100 * np.log10(sum_atr_14 / range_14) / np.log10(14)
    
    # Align 1d indicators to 4h timeframe
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Donchian channels on 4h data
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(vol_avg_aligned[i]) or 
            np.isnan(chop_aligned[i]) or
            np.isnan(donchian_high_20[i]) or
            np.isnan(donchian_low_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-day average
        volume_spike = prices['volume'].iloc[i] > (1.5 * vol_avg_aligned[i])
        
        # Trend condition: Choppiness Index < 38.2 (trending market)
        trending = chop_aligned[i] < 38.2
        
        # Range condition: Choppiness Index > 61.8 (ranging market) - used for exit
        ranging = chop_aligned[i] > 61.8
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high_20[i]
        short_breakout = close[i] < donchian_low_20[i]
        
        # Exit conditions
        exit_long = position == 1 and (close[i] < donchian_low_10[i] or not volume_spike or ranging)
        exit_short = position == -1 and (close[i] > donchian_high_10[i] or not volume_spike or ranging)
        
        # Entry conditions: breakout + volume spike + trending
        long_entry = long_breakout and volume_spike and trending
        short_entry = short_breakout and volume_spike and trending
        
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

name = "4h_1d_donchian_volume_chop"
timeframe = "4h"
leverage = 1.0