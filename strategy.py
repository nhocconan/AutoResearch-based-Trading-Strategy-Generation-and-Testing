#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian breakout + 1d EMA(50) trend + volume spike + session filter (08-20 UTC)
    # Long when: price breaks above 4h Donchian upper (20) AND price > 1d EMA50 AND volume > 1.5x 20-bar avg volume AND session 08-20 UTC
    # Short when: price breaks below 4h Donchian lower (20) AND price < 1d EMA50 AND volume > 1.5x 20-bar avg volume AND session 08-20 UTC
    # Exit when: price crosses 4h Donchian midpoint OR adverse 1d EMA50 crossover
    # Uses discrete sizing (0.20) targeting 60-150 total trades over 4 years.
    # Donchian channels provide trend-following structure; 1d EMA50 filters counter-trend moves;
    # Volume spike confirms breakout validity; session filter reduces noise trades.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    midpoint_4h = (upper_4h + lower_4h) / 2.0
    
    # Calculate 1d EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1h timeframe (wait for completed HTF bar)
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    midpoint_4h_aligned = align_htf_to_ltf(prices, df_4h, midpoint_4h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume confirmation: volume > 1.5x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% position size
    
    for i in range(20, n):
        # Skip if data not ready or outside session
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or np.isnan(midpoint_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_volume[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions (using previous bar's levels to avoid look-ahead)
        breakout_up = close[i] > upper_4h_aligned[i-1]  # break above previous upper
        breakout_down = close[i] < lower_4h_aligned[i-1]  # break below previous lower
        
        # 1d EMA50 trend filter
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions with volume confirmation and session filter
        long_entry = breakout_up and uptrend and volume_confirmed[i] and position != 1
        short_entry = breakout_down and downtrend and volume_confirmed[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and (close[i] < midpoint_4h_aligned[i] or not uptrend))
        exit_short = (position == -1 and (close[i] > midpoint_4h_aligned[i] or not downtrend))
        
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

name = "4h_1d_donchian_breakout_ema_volume_session_v1"
timeframe = "1h"
leverage = 1.0