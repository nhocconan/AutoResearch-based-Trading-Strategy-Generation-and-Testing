#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout + 1d EMA(50) trend filter + volume confirmation
    # Long when: price breaks above 6h Donchian upper (20) AND 1d close > 1d EMA(50) AND volume > 2x 20-bar avg
    # Short when: price breaks below 6h Donchian lower (20) AND 1d close < 1d EMA(50) AND volume > 2x 20-bar avg
    # Exit when: price crosses 6h Donchian midpoint
    # Uses discrete sizing (0.25) targeting 50-150 total trades over 4 years (12-37/year).
    # EMA filter ensures breakouts align with higher timeframe trend, reducing false breakouts.
    # Volume confirmation ensures breakout has participation.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Donchian channels (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate 6h Donchian channels (20-period)
    donchian_window = 20
    donchian_high_6h = pd.Series(high_6h).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low_6h = pd.Series(low_6h).rolling(window=donchian_window, min_periods=donchian_window).min().values
    donchian_mid_6h = (donchian_high_6h + donchian_low_6h) / 2.0
    
    # Align 6h Donchian levels to 15m timeframe (but we're on 6h, so this is identity)
    # Since primary timeframe is 6h, we can use the values directly
    donchian_high_aligned = donchian_high_6h
    donchian_low_aligned = donchian_low_6h
    donchian_mid_aligned = donchian_mid_6h
    
    # Get 1d data for EMA trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume confirmation: volume > 2x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(donchian_window, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(donchian_mid_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions (using current bar's close vs previous bar's levels)
        breakout_up = close[i] > donchian_high_aligned[i-1]  # break above previous Donchian high
        breakout_down = close[i] < donchian_low_aligned[i-1]  # break below previous Donchian low
        
        # Entry conditions with EMA trend filter and volume confirmation
        long_entry = breakout_up and (close_1d_aligned := close_1d[i // 24]) > ema_50_1d_aligned[i] and volume_confirmed[i] and position != 1
        short_entry = breakout_down and (close_1d_aligned := close_1d[i // 24]) < ema_50_1d_aligned[i] and volume_confirmed[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and close[i] < donchian_mid_aligned[i])
        exit_short = (position == -1 and close[i] > donchian_mid_aligned[i])
        
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

name = "6h_1d_donchian_ema_volume_filter_v1"
timeframe = "6h"
leverage = 1.0