#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout with 1d volume spike and chop regime filter
    # Enter long when price breaks above 20-period high with volume > 1.5x 20-bar avg and chop < 61.8 (trending)
    # Enter short when price breaks below 20-period low with volume > 1.5x 20-bar avg and chop < 61.8
    # Exit on opposite Donchian break or when chop > 61.8 (range) to avoid whipsaw
    # Uses 1d HTF for Donchian channels (more stable) and 12h for entry timing
    # Volume confirmation ensures breakouts have participation
    # Chop filter avoids false breakouts in ranging markets
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for primary timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data for Donchian channels (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    # Upper = max(high, 20), Lower = min(low, 20)
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation: volume > 1.5x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    # Choppiness Index (CHOP) regime filter - using 1d data
    # CHOP = 100 * log10(sum(ATR, 14) / (log10(highest high - lowest low, 14)) / log10(14))
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    # We'll use: only allow breakouts when CHOP < 61.8 (not strongly ranging)
    tr_1d = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(
            np.abs(high_1d[1:] - close_1d[:-1]),
            np.abs(low_1d[1:] - close_1d[:-1])
        )
    )
    # Prepend first TR as 0 for alignment
    tr_1d = np.concatenate([[0], tr_1d])
    atr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denominator = highest_high_14 - lowest_low_14
    # Avoid division by zero
    chop_denominator = np.where(chop_denominator == 0, 1, chop_denominator)
    chop = 100 * np.log10(atr_14 * 14 / chop_denominator) / np.log10(14)
    chop = np.where(chop_denominator == 0, 50, chop)  # neutral when range=0
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(1, n):  # start from 1 to access previous bar
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(avg_volume[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high_aligned[i]  # break above upper channel
        breakout_down = close[i] < donchian_low_aligned[i]  # break below lower channel
        
        # Regime filter: only allow breakouts when not strongly ranging (CHOP < 61.8)
        regime_filter = chop_aligned[i] < 61.8
        
        # Entry conditions with volume confirmation and regime filter
        long_entry = breakout_up and volume_confirmed[i] and regime_filter and position != 1
        short_entry = breakout_down and volume_confirmed[i] and regime_filter and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and close[i] < donchian_low_aligned[i])  # break below lower channel
        exit_short = (position == -1 and close[i] > donchian_high_aligned[i])  # break above upper channel
        regime_exit = (position != 0 and chop_aligned[i] >= 61.8)  # exit if market becomes strongly ranging
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and (exit_long or regime_exit):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (exit_short or regime_exit):
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

name = "12h_1d_donchian_volume_chop_regime_v1"
timeframe = "12h"
leverage = 1.0