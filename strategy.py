#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray + 1d ADX regime filter
    # Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
    # Long when Bull Power > 0 and Bear Power < 0 and ADX(1d) > 25 (trending)
    # Short when Bear Power > 0 and Bull Power < 0 and ADX(1d) > 25 (trending)
    # Exit when power values converge (|Bull Power| < threshold and |Bear Power| < threshold)
    # Uses 1d HTF for ADX regime (trending vs ranging) and 6h for Elder Ray timing
    # Works in bull (strong Bull Power) and bear (strong Bear Power) markets
    # ADX filter avoids whipsaws in ranging markets
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 6h data for primary timeframe
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Get 1d data for ADX regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA for Elder Ray (6h)
    close_6h_series = pd.Series(close_6h)
    ema_13 = close_6h_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components (6h)
    bull_power = high_6h - ema_13  # Bull Power = High - EMA
    bear_power = ema_13 - low_6h   # Bear Power = EMA - Low
    
    # Calculate ADX for regime filter (1d)
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d - np.roll(high_1d, 1))
    down_move = pd.Series(np.roll(low_1d, 1) - low_1d)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume filter (optional confirmation)
    volume = prices['volume'].values
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    power_threshold = 0.0  # convergence threshold for exit
    
    for i in range(1, n):  # start from 1 to access previous bar
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Regime: only trade when ADX > 25 (trending market)
        is_trending = adx_aligned[i] > 25.0
        
        # Elder Ray signals
        strong_bull = bull_power[i] > 0 and bear_power[i] < 0  # bulls in control
        strong_bear = bear_power[i] > 0 and bull_power[i] < 0  # bears in control
        converging = (abs(bull_power[i]) < power_threshold and 
                     abs(bear_power[i]) < power_threshold)  # power converging
        
        # Entry conditions
        long_entry = strong_bull and is_trending and volume_confirmed[i] and position != 1
        short_entry = strong_bear and is_trending and volume_confirmed[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and converging)
        exit_short = (position == -1 and converging)
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
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

name = "6h_1d_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0