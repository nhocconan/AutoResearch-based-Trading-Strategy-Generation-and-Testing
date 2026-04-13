#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 1d 20-period Donchian breakout with 1w EMA trend filter and volume confirmation
    # Long: close > Donchian high(20) + volume > 1.5x 20-period average + 1w EMA50 up
    # Short: close < Donchian low(20) + volume > 1.5x 20-period average + 1w EMA50 down
    # Uses discrete sizing (0.25) to minimize fee drag
    # Target: 20-25 trades/year to stay within 1d optimal range (80-100 total over 4 years)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate 1w EMA50
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 1d timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume average for confirmation (using 1d data)
    vol_avg_20_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_avg_20_1d[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_avg_20_1d[i]
        
        # Trend filter: 1w EMA50 direction (using slope)
        ema_now = ema_50_1w_aligned[i]
        ema_prev = ema_50_1w_aligned[i-1] if i > 0 else ema_now
        ema_rising = ema_now > ema_prev
        ema_falling = ema_now < ema_prev
        
        # Breakout conditions: price breaks Donchian levels with volume and trend
        breakout_long = (close[i] > donchian_high_aligned[i]) and volume_confirmed and ema_rising
        breakout_short = (close[i] < donchian_low_aligned[i]) and volume_confirmed and ema_falling
        
        # Exit conditions: reverse signal or loss of trend
        exit_long = position == 1 and (breakout_short or not ema_rising)
        exit_short = position == -1 and (breakout_long or not ema_falling)
        
        # Execute signals
        if breakout_long and position != 1:
            position = 1
            signals[i] = position_size
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long:
            position = 0
            signals[i] = 0.0
        elif exit_short:
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

name = "1d_20_donchian_weekly_trend_volume_v1"
timeframe = "1d"
leverage = 1.0