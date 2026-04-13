#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout + weekly Camarilla pivot continuation + volume confirmation
    # Long when: price breaks above 6h Donchian upper (20) AND weekly S3 pivot level holds AND volume > 2.0x 20-bar avg
    # Short when: price breaks below 6h Donchian lower (20) AND weekly R3 pivot level holds AND volume > 2.0x 20-bar avg
    # Exit when: price crosses 6h Donchian midpoint
    # Uses discrete sizing (0.25) targeting 50-150 total trades over 4 years.
    # Weekly Camarilla pivot (from 1w) provides strong institutional support/resistance levels.
    # Volume confirmation ensures breakouts have conviction.
    # Works in bull (breakouts above weekly S3 with trend) and bear (breakdowns below weekly R3 with trend).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly Camarilla pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla pivot levels (using previous week's OHLC)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # S3 = Pivot - Range * 1.1000
    # R3 = Pivot + Range * 1.1000
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    s3_1w = pivot_1w - (range_1w * 1.1000)
    r3_1w = pivot_1w + (range_1w * 1.1000)
    
    # Align weekly levels to 6h timeframe (wait for weekly close)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    
    # Calculate 6h Donchian channels (20-period)
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate volume confirmation: volume > 2.0x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(donchian_window, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(s3_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions (using current bar's close vs previous bar's levels)
        breakout_up = close[i] > donchian_high[i-1]  # break above previous Donchian high
        breakout_down = close[i] < donchian_low[i-1]  # break below previous Donchian low
        
        # Entry conditions with weekly pivot filter and volume confirmation
        # Long: breakout above Donchian AND price above weekly S3 (support holds)
        # Short: breakout below Donchian AND price below weekly R3 (resistance holds)
        long_entry = breakout_up and (close[i] > s3_1w_aligned[i]) and volume_confirmed[i] and position != 1
        short_entry = breakout_down and (close[i] < r3_1w_aligned[i]) and volume_confirmed[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and close[i] < donchian_mid[i])
        exit_short = (position == -1 and close[i] > donchian_mid[i])
        
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

name = "6h_1w_donchian_camarilla_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0