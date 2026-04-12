#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
    # Uses 1w for signal direction (long-term trend), 1d for precise entry timing
    # Volume spike (>1.8x 20-period average) confirms institutional participation
    # Only trades with the dominant 1w trend to avoid counter-trend whipsaws
    # Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Donchian calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate previous 1w bar's Donchian channels (20-period)
    # Upper = max(high_prev_20), Lower = min(low_prev_20)
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Shift by 1 to use previous completed bar
    upper_channel = np.roll(high_20, 1)
    lower_channel = np.roll(low_20, 1)
    upper_channel[0] = np.nan
    lower_channel[0] = np.nan
    
    # Get 1w EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 1d volume for confirmation (>1.8x 20-period average)
    vol_ma_1d = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_1d[i] = np.mean(volume[i-20:i])
    volume_spike_1d = volume > (1.8 * vol_ma_1d)
    
    # Align all indicators to LTF (1d)
    upper_channel_aligned = align_htf_to_ltf(prices, df_1w, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_1w, lower_channel)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper_channel_aligned[i]) or np.isnan(lower_channel_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_spike_1d[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > upper_channel_aligned[i]
        short_breakout = close[i] < lower_channel_aligned[i]
        
        # 1w trend filter
        bullish_trend = close[i] > ema34_1w_aligned[i]
        bearish_trend = close[i] < ema34_1w_aligned[i]
        
        # Entry logic: Breakout + trend alignment + volume confirmation
        long_entry = long_breakout and bullish_trend and volume_spike_1d[i]
        short_entry = short_breakout and bearish_trend and volume_spike_1d[i]
        
        # Exit logic: opposite breakout or trend reversal
        long_exit = short_breakout or not bullish_trend
        short_exit = long_breakout or not bearish_trend
        
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

name = "1d_1w_donchian_breakout_ema34_volume_v1"
timeframe = "1d"
leverage = 1.0