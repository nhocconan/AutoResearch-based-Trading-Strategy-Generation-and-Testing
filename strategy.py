#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and 12h ADX trend filter.
    # Long when price breaks above Donchian upper band + volume spike (>1.5x 20-period avg 12h volume) + ADX(14) > 25 (strong trend).
    # Short when price breaks below Donchian lower band + volume spike + ADX(14) > 25.
    # Exit when price crosses back below Donchian middle (long) or above Donchian middle (short).
    # Uses Donchian for structure, volume for confirmation, ADX to avoid whipsaws in ranging markets.
    # Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Get 12h data for volume and ADX confirmation (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate volume moving average (20-period) on 12h
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate True Range (TR) on 12h for ADX
    tr1 = np.abs(high_12h - low_12h)
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Calculate ATR (14) on 12h
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate +DM and -DM on 12h
    up_move = np.diff(high_12h, prepend=high_12h[0])
    down_move = -np.diff(low_12h, prepend=low_12h[0])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Calculate +DI and -DI (14) on 12h
    plus_di_12h = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / np.maximum(atr_12h, 1e-10)
    minus_di_12h = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / np.maximum(atr_12h, 1e-10)
    
    # Calculate DX and ADX (14) on 12h
    dx = 100 * np.abs(plus_di_12h - minus_di_12h) / np.maximum(plus_di_12h + minus_di_12h, 1e-10)
    adx_12h = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF indicators to 4h timeframe
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_12h_aligned[i]) or np.isnan(adx_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        volume_spike = volume_12h_aligned[i] > 1.5 * vol_ma_12h_aligned[i]
        
        # Trend filter: ADX > 25 indicates strong trend (good for breakouts)
        trend_filter = adx_12h_aligned[i] > 25
        
        # Breakout conditions
        long_breakout = close[i] > highest_high[i]
        short_breakout = close[i] < lowest_low[i]
        
        # Exit conditions: price crosses back below/above Donchian middle
        long_exit = close[i] < donchian_mid[i]
        short_exit = close[i] > donchian_mid[i]
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.25
        
        # Entry conditions
        if long_breakout and volume_spike and trend_filter and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and volume_spike and trend_filter and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
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

name = "4h_12h_donchian_volume_adx_v1"
timeframe = "4h"
leverage = 1.0