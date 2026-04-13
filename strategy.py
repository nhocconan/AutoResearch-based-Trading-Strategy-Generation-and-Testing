#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian breakout + 1d EMA200 trend filter + volume confirmation + session filter
    # Long when: price breaks above 4h Donchian upper (20) AND price > 1d EMA200 AND volume > 2x 20-bar avg AND 08-20 UTC
    # Short when: price breaks below 4h Donchian lower (20) AND price < 1d EMA200 AND volume > 2x 20-bar avg AND 08-20 UTC
    # Exit when: price crosses 4h Donchian midpoint
    # Uses discrete sizing (0.20) targeting 60-150 total trades over 4 years.
    # 1d EMA200 provides strong trend filter, reducing whipsaw in choppy markets and bear rallies.
    # Volume confirmation ensures breakouts have conviction (2x avg volume).
    # Session filter (08-20 UTC) reduces noise trades during low-liquidity hours.
    # Works in bull (breakouts with trend) and bear (strong trend-aligned breaks only).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Get 4h data for Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid_4h = (donchian_high_4h + donchian_low_4h) / 2.0
    
    # Align 4h Donchian levels to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid_4h)
    
    # Calculate volume confirmation: volume > 2.0x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (2.0 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% position size
    
    # Start loop after warmup period
    start_idx = max(20, 200)  # Donchian 20 + EMA200 warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(avg_volume[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions (using current bar's close vs previous bar's levels)
        breakout_up = close[i] > donchian_high_aligned[i-1]  # break above previous Donchian high
        breakout_down = close[i] < donchian_low_aligned[i-1]  # break below previous Donchian low
        
        # Entry conditions with trend filter and volume confirmation
        long_entry = breakout_up and (close[i] > ema_200_1d_aligned[i]) and volume_confirmed[i] and position != 1
        short_entry = breakout_down and (close[i] < ema_200_1d_aligned[i]) and volume_confirmed[i] and position != -1
        
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

name = "4h_1d_donchian_ema200_volume_session_v1"
timeframe = "1h"
leverage = 1.0