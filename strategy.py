#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R (14) + 1d ADX (14) regime filter + volume spike confirmation.
# Long when Williams %R < -80 (oversold) AND 1d ADX > 25 (trending) AND volume > 2.0x 20-period average.
# Short when Williams %R > -20 (overbought) AND 1d ADX > 25 (trending) AND volume > 2.0x 20-period average.
# Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts).
# Uses discrete position size 0.25. Williams %R identifies extremes, 1d ADX ensures we trade with the trend
# (avoiding chop), volume confirms momentum. Designed for 6h timeframe to capture swings in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: ADX(14) for trend strength ===
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di_14 = 100 * plus_dm_14 / tr_14
    minus_di_14 = 100 * minus_dm_14 / tr_14
    
    # ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_14 = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Get 6h data for Williams %R and volume
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Williams %R(14) on 6h
    highest_high_14 = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_6h) / (highest_high_14 - lowest_low_14)
    
    # Align Williams %R to 6h timeframe (no additional delay needed as it's based on completed bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # Volume moving average (20-period) on 6h
    vol_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20_6h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        wr = williams_r_aligned[i]
        adx = adx_aligned[i]
        vol_ma = vol_ma_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R crosses above -50
            if wr > -50:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R crosses below -50
            if wr < -50:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume filter: volume > 2.0x 20-period average
            vol_filter = vol > 2.0 * vol_ma
            
            # Trend filter: 1d ADX > 25 (trending market)
            trend_filter = adx > 25
            
            # LONG: Williams %R oversold (< -80) with volume and trend confirmation
            if wr < -80 and vol_filter and trend_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Williams %R overbought (> -20) with volume and trend confirmation
            elif wr > -20 and vol_filter and trend_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_WilliamsR14_1dADX14_VolumeFilter_V1"
timeframe = "6h"
leverage = 1.0