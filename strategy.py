#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation (>1.5x 20-period average)
    # Uses 1d EMA50 for primary trend direction, 12h for Donchian calculation and entry/exit
    # Volume spike confirms institutional participation
    # Only trades with the dominant 1d trend to avoid counter-trend whipsaws
    # Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate previous 12h bar's Donchian channels (20-period)
    # Upper = max(high_12h[-20:])
    # Lower = min(low_12h[-20:])
    lookback = 20
    upper_12h = np.full_like(high_12h, np.nan)
    lower_12h = np.full_like(low_12h, np.nan)
    
    for i in range(lookback, len(high_12h)):
        upper_12h[i] = np.max(high_12h[i-lookback:i])
        lower_12h[i] = np.min(low_12h[i-lookback:i])
    
    # Get 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 12h volume for confirmation (>1.5x 20-period average)
    vol_ma_12h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_12h[i] = np.mean(volume[i-20:i])
    volume_spike_12h = volume > (1.5 * vol_ma_12h)
    
    # Align all indicators to LTF (12h)
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper_12h_aligned[i]) or np.isnan(lower_12h_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_spike_12h[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > upper_12h_aligned[i]
        short_breakout = close[i] < lower_12h_aligned[i]
        
        # 1d trend filter
        bullish_trend = close[i] > ema50_1d_aligned[i]
        bearish_trend = close[i] < ema50_1d_aligned[i]
        
        # Entry logic: Breakout + trend alignment + volume confirmation
        long_entry = long_breakout and bullish_trend and volume_spike_12h[i]
        short_entry = short_breakout and bearish_trend and volume_spike_12h[i]
        
        # Exit logic: opposite Donchian breakout (mean reversion)
        long_exit = short_breakout  # Exit long when price breaks below lower band
        short_exit = long_breakout  # Exit short when price breaks above upper band
        
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

name = "12h_1d_donchian_breakout_ema50_volume_v1"
timeframe = "12h"
leverage = 1.0