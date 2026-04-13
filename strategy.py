#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + 1d volume spike + chop regime filter
    # Long when price breaks above Donchian(20) high AND 1d volume > 1.5x 20-day average AND chop > 61.8 (range)
    # Short when price breaks below Donchian(20) low AND 1d volume > 1.5x 20-day average AND chop > 61.8 (range)
    # Exit when price touches opposite Donchian band or regime shifts to trending (chop < 38.2)
    # Uses discrete sizing (0.25) targeting 75-200 total trades over 4 years.
    # Works in bull/bear via chop regime filter ensuring we only trade in ranging markets where breakouts fade.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for regime filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d 20-day average volume for spike detection
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d choppiness index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(1)) / (n * (HHV - LLV))) / log10(n)
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]), np.abs(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])  # align with index
    atr1 = pd.Series(tr1).rolling(window=1, min_periods=1).mean().values  # ATR(1) = TR
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    hhvl = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    llvl = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = 14 * np.maximum(hhvl - llvl, 1e-10)
    chop_raw = 100 * np.log10(sum_atr1 / chop_denom) / np.log10(14)
    chop_1d = np.where(chop_denom > 0, chop_raw, 50.0)  # default to 50 when range=0
    
    # Align 1d indicators to 4h timeframe
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma20_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or
            np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current 1d volume > 1.5x 20-day average
        volume_spike = volume_1d_aligned[i] > 1.5 * vol_ma20_1d_aligned[i]
        
        # Chop regime condition: chop > 61.8 = ranging market (fade breakouts)
        chop_range = chop_1d_aligned[i] > 61.8
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high[i]
        breakout_down = close[i] < lowest_low[i]
        
        # Entry conditions: breakout + volume spike + chop regime (range)
        long_signal = breakout_up and volume_spike and chop_range
        short_signal = breakout_down and volume_spike and chop_range
        
        # Exit conditions: touch opposite band OR regime shifts to trending (chop < 38.2)
        exit_long = close[i] < lowest_low[i] or chop_1d_aligned[i] < 38.2
        exit_short = close[i] > highest_high[i] or chop_1d_aligned[i] < 38.2
        
        # Entry logic
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit logic
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