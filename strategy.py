#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and chop regime filter
    # Long: price breaks above Donchian high (20) + volume > 1.3x 20-period 12h avg + chop > 61.8 (range)
    # Short: price breaks below Donchian low (20) + volume > 1.3x 20-period 12h avg + chop > 61.8 (range)
    # Uses discrete sizing (0.25) to minimize fee drag and ATR-based stoploss
    # Target: 12-37 trades/year to stay within 12h optimal range (50-150 total over 4 years)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation and chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ATR for chop calculation (using true range)
    atr_1d = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        tr = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
        if i < 14:
            atr_1d[i] = np.mean(atr_1d[:i+1]) if i > 0 else tr
        else:
            atr_1d[i] = 0.93 * atr_1d[i-1] + 0.07 * tr  # Wilder's smoothing
    
    # Calculate 1d Chopiness Index (14-period)
    chop_1d = np.zeros(len(close_1d))
    sum_tr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_h_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_l_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_h_14 - min_l_14
    chop_1d = 100 * np.log10(sum_tr_14 / np.log(14) / range_14)
    chop_1d = np.where(range_14 == 0, 50, chop_1d)  # Avoid division by zero
    
    # Calculate 1d volume average for confirmation
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    entry_price = np.full(n, np.nan)
    
    # Calculate 12h ATR for stoploss
    atr_12h = np.zeros(n)
    for i in range(1, n):
        tr = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        if i < 14:
            atr_12h[i] = np.mean(atr_12h[:i+1]) if i > 0 else tr
        else:
            atr_12h[i] = 0.93 * atr_12h[i-1] + 0.07 * tr
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.3x 20-period 1d average (aligned)
        volume_confirmed = volume[i] > 1.3 * vol_avg_20_1d_aligned[i]
        
        # Chop regime filter: chop > 61.8 indicates ranging market (good for mean reversion at extremes)
        chop_filter = chop_1d_aligned[i] > 61.8
        
        # Breakout conditions: price breaks Donchian levels with volume and chop filter
        breakout_long = (close[i] > donchian_high_aligned[i]) and volume_confirmed and chop_filter
        breakout_short = (close[i] < donchian_low_aligned[i]) and volume_confirmed and chop_filter
        
        # Stoploss: 2x ATR below/above entry
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - 2.0 * atr_12h[i]
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + 2.0 * atr_12h[i]
        
        # Execute signals
        if breakout_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "12h_1d_donchian_volume_chop_v3"
timeframe = "12h"
leverage = 1.0