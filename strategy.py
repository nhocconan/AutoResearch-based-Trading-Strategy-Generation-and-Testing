#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + 12h Supertrend(10,3.0) trend filter + volume confirmation
    # Long when: price breaks above 4h Donchian upper (20) AND 12h Supertrend = uptrend AND volume > 2.0x 20-bar avg
    # Short when: price breaks below 4h Donchian lower (20) AND 12h Supertrend = downtrend AND volume > 2.0x 20-bar avg
    # Exit when: price crosses 4h Donchian midpoint
    # Uses discrete sizing (0.25) targeting 75-200 total trades over 4 years (19-50/year).
    # Supertrend on 12h provides strong trend filter to avoid counter-trend breakouts.
    # Volume confirmation ensures breakout has conviction.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_window = 20
    donchian_high_4h = pd.Series(high_4h).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low_4h = pd.Series(low_4h).rolling(window=donchian_window, min_periods=donchian_window).min().values
    donchian_mid_4h = (donchian_high_4h + donchian_low_4h) / 2.0
    
    # Align 4h Donchian levels to 15m timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid_4h)
    
    # Get 12h data for Supertrend (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ATR for Supertrend (period=10)
    tr1 = np.abs(high_12h - low_12h)
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate Supertrend (10, 3.0)
    hl2_12h = (high_12h + low_12h) / 2.0
    upper_band_12h = hl2_12h + (3.0 * atr_12h)
    lower_band_12h = hl2_12h - (3.0 * atr_12h)
    
    # Initialize Supertrend arrays
    supertrend_12h = np.full_like(close_12h, np.nan)
    direction_12h = np.full_like(close_12h, np.nan)  # 1 for uptrend, -1 for downtrend
    
    # Calculate Supertrend
    for i in range(len(close_12h)):
        if i == 0:
            supertrend_12h[i] = hl2_12h[i]
            direction_12h[i] = 1  # start with uptrend
        else:
            if close_12h[i-1] > supertrend_12h[i-1]:
                # Previous close was above previous Supertrend
                upper_band_12h[i] = min(upper_band_12h[i], upper_band_12h[i-1])
                if close_12h[i] <= upper_band_12h[i]:
                    direction_12h[i] = -1
                    supertrend_12h[i] = upper_band_12h[i]
                else:
                    direction_12h[i] = 1
                    supertrend_12h[i] = upper_band_12h[i]
            else:
                # Previous close was below previous Supertrend
                lower_band_12h[i] = max(lower_band_12h[i], lower_band_12h[i-1])
                if close_12h[i] >= lower_band_12h[i]:
                    direction_12h[i] = 1
                    supertrend_12h[i] = lower_band_12h[i]
                else:
                    direction_12h[i] = -1
                    supertrend_12h[i] = lower_band_12h[i]
    
    # Align 12h Supertrend direction to 4h timeframe
    direction_12h_aligned = align_htf_to_ltf(prices, df_12h, direction_12h)
    
    # Calculate volume confirmation: volume > 2.0x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(donchian_window, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(donchian_mid_aligned[i]) or
            np.isnan(direction_12h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions (using current bar's close vs previous bar's levels)
        breakout_up = close[i] > donchian_high_aligned[i-1]  # break above previous Donchian high
        breakout_down = close[i] < donchian_low_aligned[i-1]  # break below previous Donchian low
        
        # Entry conditions with trend filter and volume confirmation
        long_entry = breakout_up and (direction_12h_aligned[i] == 1) and volume_confirmed[i] and position != 1
        short_entry = breakout_down and (direction_12h_aligned[i] == -1) and volume_confirmed[i] and position != -1
        
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

name = "4h_12h_donchian_supertrend_volume_v1"
timeframe = "4h"
leverage = 1.0