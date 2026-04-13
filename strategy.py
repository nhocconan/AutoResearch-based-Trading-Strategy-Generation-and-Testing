#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + 12h Supertrend(ATR=10,mult=3) + volume confirmation
    # Long when: price breaks above Donchian upper (20) AND 12h Supertrend = uptrend AND volume > 2.0x 20-bar avg volume
    # Short when: price breaks below Donchian lower (20) AND 12h Supertrend = downtrend AND volume > 2.0x 20-bar avg volume
    # Exit when: price crosses Donchian midpoint OR adverse 12h Supertrend crossover
    # Uses discrete sizing (0.25) targeting 75-200 trades over 4 years.
    # Donchian breakout captures momentum; 12h Supertrend filters counter-trend moves; volume reduces false breakouts.
    # Works in bull (trend-following breaks) and bear (mean-reversion exits at midpoint).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Supertrend(10,3) trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ATR(10) for 12h
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12h = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate Supertrend(10,3)
    hl2 = (high_12h + low_12h) / 2
    upper_band = hl2 + (3 * atr_12h)
    lower_band = hl2 - (3 * atr_12h)
    
    # Initialize Supertrend arrays
    supertrend = np.full_like(close_12h, np.nan)
    direction = np.full_like(close_12h, np.nan)  # 1 for uptrend, -1 for downtrend
    
    # Set first valid value
    for i in range(10, len(close_12h)):
        if np.isnan(atr_12h[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
            continue
            
        if i == 10:
            supertrend[i] = upper_band[i]
            direction[i] = -1  # start in downtrend waiting for break
        else:
            # Upper band logic
            if upper_band[i] < supertrend[i-1] or close_12h[i-1] > supertrend[i-1]:
                upper_band[i] = upper_band[i]
            else:
                upper_band[i] = supertrend[i-1]
            
            # Lower band logic
            if lower_band[i] > supertrend[i-1] or close_12h[i-1] < supertrend[i-1]:
                lower_band[i] = lower_band[i]
            else:
                lower_band[i] = supertrend[i-1]
            
            # Supertrend logic
            if direction[i-1] == -1:
                if close_12h[i] > upper_band[i]:
                    supertrend[i] = lower_band[i]
                    direction[i] = 1
                else:
                    supertrend[i] = upper_band[i]
                    direction[i] = -1
            else:
                if close_12h[i] < lower_band[i]:
                    supertrend[i] = upper_band[i]
                    direction[i] = -1
                else:
                    supertrend[i] = lower_band[i]
                    direction[i] = 1
    
    # Align Supertrend direction to 4h timeframe
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_12h, direction)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate volume confirmation: volume > 2.0x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(supertrend_direction_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_upper[i-1]  # break above previous upper
        breakout_down = close[i] < donchian_lower[i-1]  # break below previous lower
        
        # 12h Supertrend trend filter
        uptrend = supertrend_direction_aligned[i] == 1
        downtrend = supertrend_direction_aligned[i] == -1
        
        # Entry conditions with volume confirmation
        long_entry = breakout_up and uptrend and volume_confirmed[i] and position != 1
        short_entry = breakout_down and downtrend and volume_confirmed[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and (close[i] < donchian_mid[i] or not uptrend))
        exit_short = (position == -1 and (close[i] > donchian_mid[i] or not downtrend))
        
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

name = "4h_12h_donchian_breakout_supertrend_volume_v1"
timeframe = "4h"
leverage = 1.0