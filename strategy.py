#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation (>1.5x 20-period average)
    # Only trade breakouts in direction of 1d trend to avoid counter-trend whipsaws
    # Volume confirmation ensures institutional participation
    # Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
    # Works in bull/bear markets by only trading with the dominant 1d trend
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate previous 1d bar's Donchian channels (20-period)
    # Upper = max(high_prev_20), Lower = min(low_prev_20)
    upper_20 = np.full(len(df_1d), np.nan)
    lower_20 = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        upper_20[i] = np.max(high_1d[i-20:i])
        lower_20[i] = np.min(low_1d[i-20:i])
    
    # Get 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d volume for confirmation (>1.5x 20-period average)
    vol_ma_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    volume_spike_1d = volume_1d > (1.5 * vol_ma_1d)
    
    # Align all indicators to LTF (6h)
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > upper_20_aligned[i]
        short_breakout = close[i] < lower_20_aligned[i]
        
        # 1d trend filter
        bullish_trend = close[i] > ema50_1d_aligned[i]
        bearish_trend = close[i] < ema50_1d_aligned[i]
        
        # Entry logic: Breakout + trend alignment + volume confirmation
        long_entry = long_breakout and bullish_trend and volume_spike_aligned[i]
        short_entry = short_breakout and bearish_trend and volume_spike_aligned[i]
        
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

name = "6h_1d_donchian_breakout_ema50_volume_v1"
timeframe = "6h"
leverage = 1.0