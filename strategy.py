#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w ADX(14) for trend strength
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr3 = np.abs(low_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1w - np.concatenate([[high_1w[0]], high_1w[:-1]])
    down_move = np.concatenate([[low_1w[0]], low_1w[:-1]]) - low_1w
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    atr_1w = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_1w = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_1w + 1e-10)
    minus_di_1w = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_1w + 1e-10)
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w + 1e-10)
    adx_1w = pd.Series(dx_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1w ADX to 6h
    adx_1w_6h = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Get 1d HTF data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    upper_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian to 6h
    upper_20_6h = align_htf_to_ltf(prices, df_1d, upper_20_1d)
    lower_20_6h = align_htf_to_ltf(prices, df_1d, lower_20_1d)
    
    # Calculate 6h ATR(14) for volatility filter
    tr1_6h = high - low
    tr2_6h = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3_6h = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr_14_6h = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_1w_6h[i]) or np.isnan(upper_20_6h[i]) or np.isnan(lower_20_6h[i]) or 
            np.isnan(atr_14_6h[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Strong weekly trend: ADX > 25
        # 2. 6h price breaks above 1d Donchian upper (20) - bullish breakout
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        if (adx_1w_6h[i] > 25 and
            close[i] > upper_20_6h[i] and
            volume_ratio[i] > 1.5 and
            atr_14_6h[i] > 0.005 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Strong weekly trend: ADX > 25
        # 2. 6h price breaks below 1d Donchian lower (20) - bearish breakdown
        # 3. Volume confirmation: volume > 1.5x average
        # 4. Volatility filter: ATR > 0.5% of price
        elif (adx_1w_6h[i] > 25 and
              close[i] < lower_20_6h[i] and
              volume_ratio[i] > 1.5 and
              atr_14_6h[i] > 0.005 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_1w_ADX25_1d_Donchian20_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0