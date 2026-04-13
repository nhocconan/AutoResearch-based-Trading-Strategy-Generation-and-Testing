#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 12h EMA trend filter and 1d volume spike.
    # Long when price breaks above Donchian high(20) AND 12h EMA(50) rising AND 1d volume > 1.8x 20-period MA.
    # Short when price breaks below Donchian low(20) AND 12h EMA(50) falling AND 1d volume > 1.8x 20-period MA.
    # Exit when price crosses Donchian midpoint OR 12h EMA(50) flips direction.
    # Uses discrete position sizing (0.25) to target 50-150 trades over 4 years.
    # Works in bull/bear via EMA trend filter avoiding counter-trend false signals.
    
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
    
    # Calculate Donchian channels (20-period)
    highest_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high_20 + lowest_low_20) / 2
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate EMA(50) on 12h
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_prev = np.roll(ema_50_12h, 1)
    ema_50_12h_prev[0] = ema_50_12h[0]
    ema_rising = ema_50_12h > ema_50_12h_prev
    ema_falling = ema_50_12h < ema_50_12h_prev
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary timeframe (prices)
    highest_high_20_aligned = align_htf_to_ltf(prices, df_4h, highest_high_20)
    lowest_low_20_aligned = align_htf_to_ltf(prices, df_4h, lowest_low_20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_rising.astype(float))
    ema_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_falling.astype(float))
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(highest_high_20_aligned[i]) or np.isnan(lowest_low_20_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.8x 20-period average
        volume_spike = volume_1d_aligned[i] > 1.8 * vol_ma_20_1d_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high_20_aligned[i]
        breakout_down = close[i] < lowest_low_20_aligned[i]
        
        # Exit conditions: price crosses Donchian midpoint OR EMA trend flips
        exit_long = close[i] < donchian_mid_aligned[i] or (position == 1 and not ema_rising_aligned[i])
        exit_short = close[i] > donchian_mid_aligned[i] or (position == -1 and not ema_falling_aligned[i])
        
        # Entry conditions
        if breakout_up and ema_rising_aligned[i] and volume_spike and position != 1:
            position = 1
            signals[i] = position_size
        elif breakout_down and ema_falling_aligned[i] and volume_spike and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
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

name = "4h_12h_1d_donchian_breakout_ema_volume_v1"
timeframe = "4h"
leverage = 1.0