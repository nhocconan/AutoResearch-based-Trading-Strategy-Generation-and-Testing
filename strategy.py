#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d ADX for trend strength + 6h Donchian(20) breakout + volume confirmation.
# Enter long when 1d ADX > 25 (trending) + price breaks above 6h Donchian upper channel + volume > 1.5x 20-bar average.
# Enter short when 1d ADX > 25 + price breaks below 6h Donchian lower channel + volume > 1.5x 20-bar average.
# Exit when price crosses back inside Donchian channel or ADX < 20 (trend weakening).
# ADX filters out ranging markets, Donchian captures breakouts, volume confirms conviction.
# Works in bull markets (upward breakouts) and bear markets (downward breakouts).
# Uses discrete position sizing (0.25) to control risk. Target: 50-150 total trades over 4 years.

name = "6h_ADX_Donchian_Breakout_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d).shift()).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = -pd.Series(low_1d).diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0))
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0))
    
    # Smoothed DM
    tr_ma = atr  # already smoothed
    plus_di = 100 * (plus_dm.rolling(window=14, min_periods=14).sum() / tr_ma)
    minus_di = 100 * (minus_dm.rolling(window=14, min_periods=14).sum() / tr_ma)
    
    # DX and ADX
    dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], 0)
    adx = dx.rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 6h Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # ADX trend filter
        strong_trend = adx_aligned[i] > 25
        weak_trend = adx_aligned[i] < 20
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_upper[i]
        breakout_down = close[i] < donchian_lower[i]
        
        # Entry conditions
        long_entry = strong_trend and breakout_up and volume_confirm[i]
        short_entry = strong_trend and breakout_down and volume_confirm[i]
        
        # Exit conditions: trend weakening or price crosses back inside channel
        long_exit = weak_trend or (close[i] < donchian_upper[i])  # exit if trend weak or price falls below upper band
        short_exit = weak_trend or (close[i] > donchian_lower[i])  # exit if trend weak or price rises above lower band
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals