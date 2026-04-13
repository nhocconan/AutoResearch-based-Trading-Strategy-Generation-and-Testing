#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + 1d EMA200 trend filter + volume confirmation (>1.5x 20-period average)
    # Uses 4h for entry timing, 1d for primary trend direction and Donchian calculation
    # Volume spike confirms institutional participation
    # Target: 19-50 trades/year (75-200 total over 4 years) to minimize fee drag
    # Only trades with the dominant 1d trend to avoid counter-trend whipsaws
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    # Upper = max(high_1d over last 20 periods)
    # Lower = min(low_1d over last 20 periods)
    donchian_upper = np.full(len(high_1d), np.nan)
    donchian_lower = np.full(len(low_1d), np.nan)
    
    for i in range(20, len(high_1d)):
        donchian_upper[i] = np.max(high_1d[i-20:i])
        donchian_lower[i] = np.min(low_1d[i-20:i])
    
    # Get 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Get 4h volume for confirmation (>1.5x 20-period average)
    vol_ma_4h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_4h[i] = np.mean(volume[i-20:i])
    volume_spike_4h = volume > (1.5 * vol_ma_4h)
    
    # Align all indicators to LTF (4h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(volume_spike_4h[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > donchian_upper_aligned[i]
        short_breakout = close[i] < donchian_lower_aligned[i]
        
        # 1d trend filter
        bullish_trend = close[i] > ema200_1d_aligned[i]
        bearish_trend = close[i] < ema200_1d_aligned[i]
        
        # Entry logic: Breakout + trend alignment + volume confirmation
        long_entry = long_breakout and bullish_trend and volume_spike_4h[i]
        short_entry = short_breakout and bearish_trend and volume_spike_4h[i]
        
        # Exit logic: opposite Donchian breakout (mean reversion)
        long_exit = short_breakout
        short_exit = long_breakout
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_breakout_ema200_volume_v1"
timeframe = "4h"
leverage = 1.0