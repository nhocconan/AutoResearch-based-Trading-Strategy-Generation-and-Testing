#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 1h 4h Donchian breakout with 1d volume confirmation
    # Long: price breaks above 4h Donchian upper + 1d volume > 1.5x 20-day average + session filter
    # Short: price breaks below 4h Donchian lower + 1d volume > 1.5x 20-day average + session filter
    # Uses discrete sizing (0.20) to minimize fee drag
    # Target: 15-37 trades/year to stay within 1h optimal range (60-150 total over 4 years)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = pd.to_datetime(prices['open_time'])
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe
    donchian_upper_1h = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_1h = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    volume_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_avg_20_1d)
    
    # Pre-compute session hours (08-20 UTC)
    hours = open_time.dt.hour.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% position size
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_1h[i]) or 
            np.isnan(donchian_lower_1h[i]) or
            np.isnan(volume_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # Volume confirmation: current 1d volume > 1.5x 20-day average
        volume_confirmed = volume_1d[i//24] > 1.5 * volume_avg_20_1d_aligned[i] if i >= 24 else False
        
        # Breakout conditions: price breaks Donchian levels with volume confirmation and session
        breakout_long = (close[i] > donchian_upper_1h[i]) and volume_confirmed and in_session
        breakout_short = (close[i] < donchian_lower_1h[i]) and volume_confirmed and in_session
        
        # Exit conditions: reverse signal or end of session
        exit_long = position == 1 and (breakout_short or not in_session)
        exit_short = position == -1 and (breakout_long or not in_session)
        
        # Execute signals
        if breakout_long and position != 1:
            position = 1
            signals[i] = position_size
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1h_4h_donchian_1d_volume_session_v1"
timeframe = "1h"
leverage = 1.0