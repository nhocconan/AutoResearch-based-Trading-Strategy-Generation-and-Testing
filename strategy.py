#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and chop regime filter
    # Long: price breaks above Donchian(20) high + 1d volume > 1.5x 20-period average + chop < 61.8 (trending)
    # Short: price breaks below Donchian(20) low + 1d volume > 1.5x 20-period average + chop < 61.8 (trending)
    # Exit: price crosses Donchian(20) midline
    # Uses 1d volume to confirm institutional interest and chop filter to avoid whipsaws in ranging markets
    # Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
    # Target: 100-200 total trades over 4 years (25-50/year) to balance opportunity and fees
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for primary timeframe
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Get 1d data for volume confirmation and chop regime (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['taker_buy_volume'].values  # proxy for volume
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian(20) channels on 4h
    highest_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_high = highest_high
    donchian_low = lowest_low
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 1d volume ratio (current vs 20-period average)
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume_1d / vol_ma, 1.0)
    
    # Calculate 1d Chopiness Index (14-period)
    def calculate_chop(high, low, close, window=14):
        atr = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))).rolling(window=window, min_periods=1).sum()
        highest_high = pd.Series(high).rolling(window=window, min_periods=1).max()
        lowest_low = pd.Series(low).rolling(window=window, min_periods=1).min()
        chop = 100 * np.log10(atr / (highest_high - lowest_low)) / np.log10(window)
        return np.where((highest_high - lowest_low) == 0, 50, chop.values)
    
    chop = calculate_chop(high_1d, low_1d, close_1d, 14)
    
    # Align 1d indicators to 4h timeframe
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # start from 20 to have enough data for Donchian
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ratio_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high[i]
        breakout_down = close[i] < donchian_low[i]
        
        # Volume and regime filters
        volume_confirmed = vol_ratio_aligned[i] > 1.5
        trending_regime = chop_aligned[i] < 61.8  # chop < 61.8 indicates trending market
        
        # Entry conditions
        long_entry = breakout_up and volume_confirmed and trending_regime and position != 1
        short_entry = breakout_down and volume_confirmed and trending_regime and position != -1
        
        # Exit conditions (midline cross)
        exit_long = position == 1 and close[i] < donchian_mid[i]
        exit_short = position == -1 and close[i] > donchian_mid[i]
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
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

name = "4h_1d_donchian_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0