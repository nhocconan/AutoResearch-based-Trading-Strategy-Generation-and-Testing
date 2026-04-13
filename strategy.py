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
    
    # Daily data for Donchian and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Daily Donchian channels (20-period)
    highest_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily ADX (14-period) for trend strength
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Weekly EMA for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all daily data to lower timeframe
    highest_high_20_aligned = align_htf_to_ltf(prices, df_1d, highest_high_20)
    lowest_low_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_low_20)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(highest_high_20_aligned[i]) or np.isnan(lowest_low_20_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > highest_high_20_aligned[i]
        short_breakout = close[i] < lowest_low_20_aligned[i]
        
        # Trend filter: only trade in direction of weekly trend
        long_trend = close[i] > ema_20_1w_aligned[i]
        short_trend = close[i] < ema_20_1w_aligned[i]
        
        # ADX filter: only trade when trend is strong enough (ADX > 25)
        strong_trend = adx_aligned[i] > 25
        
        # Exit conditions
        long_exit = close[i] < lowest_low_20_aligned[i]  # Exit on opposite break
        short_exit = close[i] > highest_high_20_aligned[i]  # Exit on opposite break
        
        if position == 0:
            if long_breakout and long_trend and strong_trend:
                position = 1
                signals[i] = position_size
            elif short_breakout and short_trend and strong_trend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            if short_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Donchian_Trend_Filter"
timeframe = "1d"
leverage = 1.0