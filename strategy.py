#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian breakout with 1d volume confirmation and chop regime filter
    # Long when price breaks above Donchian(20) high + 1d volume > 1.5x average + chop < 61.8
    # Short when price breaks below Donchian(20) low + 1d volume > 1.5x average + chop < 61.8
    # Uses discrete position sizing (0.25) to minimize fee churn
    # Target: 75-200 total trades over 4 years (~19-50/year)
    # Donchian captures breakouts; volume confirms conviction; chop filter avoids ranging markets
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for volume and chop (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d volume average (20-period) with min_periods
    volume_1d = df_1d['volume'].values
    volume_series = pd.Series(volume_1d)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 1d chop regime (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr.rolling(window=14, min_periods=14).sum() / 
                           np.log10(highest_high - lowest_low)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) == 0, 50, chop)  # avoid division by zero
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 4h Donchian channels (20-period) with min_periods
    highest_high_4h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_4h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(vol_ma_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(highest_high_4h[i]) or np.isnan(lowest_low_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5 * 20-period average
        vol_1d_current = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_current)
        volume_confirm = vol_1d_aligned[i] > 1.5 * vol_ma_aligned[i]
        
        # Chop regime filter: chop < 61.8 (trending market)
        chop_filter = chop_aligned[i] < 61.8
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high_4h[i-1]  # break above previous high
        breakout_down = close[i] < lowest_low_4h[i-1]  # break below previous low
        
        # Entry signals
        long_entry = breakout_up and volume_confirm and chop_filter
        short_entry = breakout_down and volume_confirm and chop_filter
        
        # Exit conditions: opposite Donchian breakout or chop > 61.8 (ranging market)
        long_exit = breakout_down or chop_aligned[i] > 61.8
        short_exit = breakout_up or chop_aligned[i] > 61.8
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0