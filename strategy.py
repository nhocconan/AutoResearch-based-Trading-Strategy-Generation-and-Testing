#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4h/1d ADX Trend + Volume Breakout with Session Filter
# Uses 4h ADX to detect strong trends, 1d volume spike to confirm breakouts, and 1h Donchian breakout for entry.
# Session filter (08-20 UTC) reduces noise. Target: 15-37 trades/year (60-150 total) to avoid fee drag.
# Works in bull markets (trend follow) and bear (mean-revert on overextended volume spikes).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h data for trend strength
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 4h ADX calculation (14-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = np.diff(high_4h, prepend=high_4h[0])
    down_move = -np.diff(low_4h, prepend=low_4h[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_dm14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm14 / tr14
    minus_di = 100 * minus_dm14 / tr14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all data to 1h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_4h, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_4h, minus_di)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # 1h Donchian channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20  # 20% position size
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(plus_di_aligned[i]) or 
            np.isnan(minus_di_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 2x daily volume MA (spike)
        volume_condition = volume[i] > (volume_ma_20_1d_aligned[i] * 2.0)
        
        # ADX trend strength (>25) and direction
        strong_trend = adx_aligned[i] > 25
        bullish = plus_di_aligned[i] > minus_di_aligned[i]
        bearish = minus_di_aligned[i] > plus_di_aligned[i]
        
        # Entry conditions
        breakout_long = close[i] > high_roll[i]
        breakout_short = close[i] < low_roll[i]
        
        if position == 0:
            if strong_trend and bullish and breakout_long and volume_condition:
                position = 1
                signals[i] = position_size
            elif strong_trend and bearish and breakout_short and volume_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: trend weakening or opposite breakout
            if adx_aligned[i] < 20 or breakout_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: trend weakening or opposite breakout
            if adx_aligned[i] < 20 or breakout_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h1d_ADX_Trend_Volume_Breakout_Session"
timeframe = "1h"
leverage = 1.0