#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h strategy using 12h Donchian breakout with 1d ADX trend filter and volume confirmation
    # Works in bull/bear: Donchian(20) captures breakouts, ADX>25 ensures trending regime,
    # volume > 1.5x average confirms momentum. Discrete sizing (0.25) minimizes fee drag.
    # Target: 20-40 trades/year to stay within 4h optimal range.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 12h data for Donchian channels (primary signal)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values if 'volume' in df_12h.columns else np.ones(len(df_12h))
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h Donchian channels (20-period)
    highest_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ADX (14-period)
    # True Range
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d)
    tr2 = pd.Series(high_1d) - pd.Series(low_1d).shift(1)
    tr3 = pd.Series(close_1d).shift(1) - pd.Series(close_1d)
    tr = pd.concat([tr1.abs(), tr2.abs(), tr3.abs()], axis=1).max(axis=1).values
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = -pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_sum = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_sum = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_sum / tr_sum
    minus_di = 100 * minus_dm_sum / tr_sum
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Get 1d volume for confirmation (20-period average)
    vol_12h = df_12h['volume'].values if 'volume' in df_12h.columns else np.ones(len(df_12h))
    vol_avg_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 4h primary timeframe
    highest_high_12h_aligned = align_htf_to_ltf(prices, df_12h, highest_high_12h)
    lowest_low_12h_aligned = align_htf_to_ltf(prices, df_12h, lowest_low_12h)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_avg_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(highest_high_12h_aligned[i]) or 
            np.isnan(lowest_low_12h_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_avg_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        idx_12h = i // (2 * 6)  # 12h bars in 4h timeframe (2 bars per 12h)
        if idx_12h >= len(volume_12h):
            signals[i] = 0.0
            continue
        volume_confirmed = volume_12h[idx_12h] > 1.5 * vol_avg_20_12h_aligned[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        # Entry conditions: Donchian breakout + trend + volume
        enter_long = (close[i] > highest_high_12h_aligned[i]) and trending and volume_confirmed
        enter_short = (close[i] < lowest_low_12h_aligned[i]) and trending and volume_confirmed
        
        # Stoploss: ATR-based using 12h true range
        if idx_12h < len(high_12h) and idx_12h < len(low_12h) and idx_12h < len(close_12h):
            tr_12h = max(
                high_12h[idx_12h] - low_12h[idx_12h],
                abs(high_12h[idx_12h] - close_12h[idx_12h - 1]) if idx_12h > 0 else 0,
                abs(low_12h[idx_12h] - close_12h[idx_12h - 1]) if idx_12h > 0 else 0
            )
        else:
            tr_12h = 0
        stop_distance = tr_12h * 2.0  # 2x ATR stoploss
        
        exit_long = position == 1 and not np.isnan(entry_price[i-1]) and close[i] < entry_price[i-1] - stop_distance
        exit_short = position == -1 and not np.isnan(entry_price[i-1]) and close[i] > entry_price[i-1] + stop_distance
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "4h_12h_1d_donchian_adx_volume_v1"
timeframe = "4h"
leverage = 1.0