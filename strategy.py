#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian(20) breakout with 1w volume confirmation and ADX regime filter
    # Enter long when price breaks above 20-day high with volume > 1.5x 20-day avg volume and ADX > 25
    # Enter short when price breaks below 20-day low with volume > 1.5x 20-day avg volume and ADX > 25
    # Exit when price crosses the 20-day midpoint (mean of high and low over 20 days)
    # Uses 1w HTF for volume confirmation (more stable than 1d) and 1d for price action
    # Volume confirmation ensures breakouts have participation
    # ADX > 25 ensures we only trade in trending markets, avoiding chop
    # Works in bull (continuation breaks) and bear (reversal breaks at extremes)
    # Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for primary timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for HTF volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    volume_1w = df_1w['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    # Highest high over 20 days
    highest_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lowest low over 20 days
    lowest_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    # Midpoint (mean of highest high and lowest low over 20 days)
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Calculate 1d ADX (14-period) for regime filter
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = np.concatenate([[0], high_1d[1:] - high_1d[:-1]])
    down_move = np.concatenate([[0], low_1d[:-1] - low_1d[1:]])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 1d timeframe (already aligned, but for consistency)
    highest_high_aligned = align_htf_to_ltf(prices, df_1d, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_1d, lowest_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: volume > 1.5x 20-day average volume (using 1w HTF for stability)
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    volume_confirmed = volume_1d > (1.5 * avg_volume_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # start from 20 to ensure indicators are ready
        # Skip if data not ready
        if (np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]) or np.isnan(donchian_mid_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(avg_volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close_1d[i] > highest_high_aligned[i]  # break above 20-day high
        breakout_down = close_1d[i] < lowest_low_aligned[i]  # break below 20-day low
        
        # Entry conditions with volume confirmation and ADX filter
        long_entry = breakout_up and volume_confirmed[i] and adx_aligned[i] > 25 and position != 1
        short_entry = breakout_down and volume_confirmed[i] and adx_aligned[i] > 25 and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and close_1d[i] < donchian_mid_aligned[i])
        exit_short = (position == -1 and close_1d[i] > donchian_mid_aligned[i])
        
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

name = "1d_1w_donchian_adx_volume_filter_v1"
timeframe = "1d"
leverage = 1.0