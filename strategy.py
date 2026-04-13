#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 20-period Donchian channels on 12h
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume on 12h
    avg_volume_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 14-period RSI on 12h
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Align indicators to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    avg_volume_aligned = align_htf_to_ltf(prices, df_12h, avg_volume_12h)
    rsi_14_aligned = align_htf_to_ltf(prices, df_12h, rsi_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(avg_volume_aligned[i]) or
            np.isnan(rsi_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average
        volume_confirm = volume_12h[i] > 1.5 * avg_volume_12h[i] if not np.isnan(avg_volume_12h[i]) else False
        
        # Breakout conditions
        breakout_up = close_12h[i] > donchian_high_aligned[i]
        breakout_down = close_12h[i] < donchian_low_aligned[i]
        
        # RSI filter: avoid extreme levels
        rsi_not_overbought = rsi_14_aligned[i] < 70
        rsi_not_oversold = rsi_14_aligned[i] > 30
        
        # Entry conditions with volume confirmation
        long_entry = breakout_up and volume_confirm and rsi_not_overbought
        short_entry = breakout_down and volume_confirm and rsi_not_oversold
        
        # Exit conditions: opposite breakout or RSI extreme
        exit_long = position == 1 and (breakout_down or rsi_14_aligned[i] > 80)
        exit_short = position == -1 and (breakout_up or rsi_14_aligned[i] < 20)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
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

name = "6h_12h_donchian_breakout_volume_rsi"
timeframe = "6h"
leverage = 1.0