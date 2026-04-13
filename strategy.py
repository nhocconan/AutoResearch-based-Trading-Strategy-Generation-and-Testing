#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h timeframe with 12h Donchian breakout + volume confirmation + ADX trend filter
# Strategy: Long when price breaks above 12h Donchian high (20) with volume > 1.5x average and ADX > 25
# Short when price breaks below 12h Donchian low (20) with volume > 1.5x average and ADX > 25
# Uses ADX to filter for trending markets only, avoiding whipsaws in ranging conditions
# Volume surge confirms breakout strength
# Target: 20-50 total trades over 4 years (5-12.5/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Donchian channels (20-period high/low)
    high_20_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate ADX (14) on 12h for trend strength
    # +DM, -DM, TR
    high_diff = np.diff(high_12h, prepend=high_12h[0])
    low_diff = -np.diff(low_12h, prepend=low_12h[0])
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    tr1 = np.abs(np.diff(high_12h, prepend=high_12h[0]))
    tr2 = np.abs(np.diff(low_12h, prepend=low_12h[0]))
    tr3 = np.abs(high_12h[1:] - low_12h[:-1])
    tr3 = np.concatenate([[tr3[0]] if len(tr3) > 0 else [0], tr3])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_12h = wilders_smoothing(tr, 14)
    plus_di_12h = 100 * wilders_smoothing(plus_dm, 14) / atr_12h
    minus_di_12h = 100 * wilders_smoothing(minus_dm, 14) / atr_12h
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h + 1e-10)
    adx_12h = wilders_smoothing(dx_12h, 14)
    
    # Volume average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h indicators to 4h timeframe
    high_20_12h_aligned = align_htf_to_ltf(prices, df_12h, high_20_12h)
    low_20_12h_aligned = align_htf_to_ltf(prices, df_12h, low_20_12h)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(high_20_12h_aligned[i]) or 
            np.isnan(low_20_12h_aligned[i]) or 
            np.isnan(adx_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume surge condition
        volume_surge = volume[i] > 1.5 * vol_ma_20[i]
        
        # Breakout conditions
        long_breakout = close[i] > high_20_12h_aligned[i]
        short_breakout = close[i] < low_20_12h_aligned[i]
        
        # Trend filter
        strong_trend = adx_12h_aligned[i] > 25
        
        # Entry logic
        long_entry = long_breakout and volume_surge and strong_trend
        short_entry = short_breakout and volume_surge and strong_trend
        
        # Exit conditions: opposite breakout or loss of trend
        exit_long = position == 1 and (short_breakout or adx_12h_aligned[i] < 20)
        exit_short = position == -1 and (long_breakout or adx_12h_aligned[i] < 20)
        
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

name = "4h_12h_donchian_breakout_volume_adx_v1"
timeframe = "4h"
leverage = 1.0