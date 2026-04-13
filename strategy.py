#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 12h ADX trend filter and volume confirmation.
    # Donchian provides objective price channels, ADX filters for trending markets only,
    # volume confirms breakout strength. Discrete sizing (0.0, ±0.25) minimizes fee churn.
    # Target: 75-200 total trades over 4 years (19-50/year).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX trend filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX(14) on 12h
    plus_dm = np.zeros_like(high_12h)
    minus_dm = np.zeros_like(low_12h)
    for i in range(1, len(high_12h)):
        up_move = high_12h[i] - high_12h[i-1]
        down_move = low_12h[i-1] - low_12h[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    tr = np.maximum(np.maximum(high_12h - low_12h, np.abs(high_12h - np.roll(close_12h, 1))), np.abs(low_12h - np.roll(close_12h, 1)))
    tr[0] = high_12h[0] - low_12h[0]  # first TR
    
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di_12h = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_12h
    minus_di_12h = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_12h
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h + 1e-10)
    adx_12h = pd.Series(dx_12h).rolling(window=14, min_periods=14).mean().values
    
    # Align 12h ADX to 4h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume MA(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx_12h_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5 * 20-period MA
        volume_filter = volume[i] > 1.5 * volume_ma[i]
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_12h_aligned[i] > 25
        
        # Breakout conditions: price breaks Donchian channels with volume and trend confirmation
        long_breakout = (close[i] > donchian_high[i-1]) and volume_filter and trending
        short_breakout = (close[i] < donchian_low[i-1]) and volume_filter and trending
        
        # Exit conditions: price returns to opposite Donchian level (mean reversion within channel)
        long_exit = close[i] < donchian_low[i-1]
        short_exit = close[i] > donchian_high[i-1]
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.25
        
        # Entry conditions
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
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

name = "4h_12h_donchian_adx_volume_v1"
timeframe = "4h"
leverage = 1.0