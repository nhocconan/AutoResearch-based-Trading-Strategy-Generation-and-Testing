#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h ADX(14) for trend strength
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_12h - np.roll(high_12h, 1)
    down_move = np.roll(low_12h, 1) - low_12h
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di_12h = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / (atr_12h + 1e-10)
    minus_di_12h = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / (atr_12h + 1e-10)
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h + 1e-10)
    adx_12h = pd.Series(dx_12h).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 6h
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate 12h Donchian(20) channels
    donch_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donch_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    
    # Calculate 6h volume filter
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 200
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_12h_aligned[i]) or 
            np.isnan(donch_high_12h_aligned[i]) or
            np.isnan(donch_low_12h_aligned[i]) or
            np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: strong trend (ADX > 25)
        trend_filter = adx_12h_aligned[i] > 25
        
        # Breakout filters
        breakout_long = price > donch_high_12h_aligned[i]
        breakout_short = price < donch_low_12h_aligned[i]
        
        # Volume confirmation: volume > 1.5x average
        vol_filter = vol > 1.5 * vol_ma_6h[i]
        
        if position == 0:
            # Long setup: bullish breakout + strong trend + volume
            if breakout_long and trend_filter and vol_filter:
                position = 1
                signals[i] = position_size
            # Short setup: bearish breakout + strong trend + volume
            elif breakout_short and trend_filter and vol_filter:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below Donchian low or trend weakens
            if price < donch_low_12h_aligned[i] or adx_12h_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above Donchian high or trend weakens
            if price > donch_high_12h_aligned[i] or adx_12h_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_12hADX_Donchian_Breakout_Volume_v1"
timeframe = "6h"
leverage = 1.0