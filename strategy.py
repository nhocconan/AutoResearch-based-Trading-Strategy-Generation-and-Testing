#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band breakout with 1d ADX trend filter and volume confirmation.
# Long when price breaks above upper BB(20,2) in 1d uptrend (ADX>25) with volume spike.
# Short when price breaks below lower BB(20,2) in 1d downtrend (ADX>25) with volume spike.
# Uses discrete sizing 0.25 to balance return and drawdown. Target: 50-150 total trades over 4 years.
# Bollinger Bands capture volatility expansion, 1d ADX>25 ensures trending market alignment,
# Volume spike confirms institutional participation. Works in both bull and bear markets by only
# trading with the 1d trend, avoiding counter-trend whipsaws. Designed for 6h timeframe to minimize fee drag.

name = "6h_BB_Breakout_1dADX25_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 6h data for Bollinger Bands calculation
    df_6h = get_htf_data(prices, '6h')
    
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate 6h Bollinger Bands (20,2)
    close_6h = df_6h['close'].values
    sma_20 = pd.Series(close_6h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_6h).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    upper_bb_aligned = align_htf_to_ltf(prices, df_6h, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_6h, lower_bb)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 1d ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d)
    tr2 = pd.Series(close_1d).shift(1) - pd.Series(high_1d)
    tr3 = pd.Series(close_1d).shift(1) - pd.Series(low_1d)
    tr = pd.concat([tr1.abs(), tr2.abs(), tr3.abs()], axis=1).max(axis=1)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = pd.Series(low_1d).diff().abs()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * (plus_dm_smooth / atr)
    minus_di = 100 * (minus_dm_smooth / atr)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike detection (20-period volume MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        vol_spike = volume_spike[i]
        adx_val = adx_aligned[i]
        is_trending = adx_val > 25  # Strong trend filter
        
        if position == 0:
            # Long: Price breaks above upper BB AND 1d trending (ADX>25) AND volume spike
            if close_val > upper_bb_aligned[i] and is_trending and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower BB AND 1d trending (ADX>25) AND volume spike
            elif close_val < lower_bb_aligned[i] and is_trending and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below middle BB (SMA20) OR ADX drops below 20 (trend weakening)
            if close_val < sma_20_aligned[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above middle BB (SMA20) OR ADX drops below 20 (trend weakening)
            if close_val > sma_20_aligned[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Pre-compute SMA20 aligned for exit conditions
df_6h_temp = get_htf_data(prices, '6h') if 'prices' in locals() else None
if df_6h_temp is not None and len(df_6h_temp) >= 20:
    close_6h_temp = df_6h_temp['close'].values
    sma_20_temp = pd.Series(close_6h_temp).rolling(window=20, min_periods=20).mean().values
    sma_20_aligned = align_htf_to_ltf(prices, df_6h_temp, sma_20_temp)
else:
    sma_20_aligned = np.full_like(close, np.nan) if 'close' in locals() else np.array([])