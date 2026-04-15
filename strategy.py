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
    
    # Get daily HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    upper_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian to daily (no shift needed as we're on 1d timeframe)
    upper_20_daily = upper_20_1d  # Already aligned since timeframe is 1d
    lower_20_daily = lower_20_1d
    
    # Get weekly HTF data for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly ADX(14) for trend strength
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr3 = np.abs(low_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.concatenate([[high_1w[0]], high_1w[:-1]])) > 
                       (np.concatenate([[low_1w[0]], low_1w[:-1]]) - low_1w), 
                       np.maximum(high_1w - np.concatenate([[high_1w[0]], high_1w[:-1]]), 0), 0)
    dm_minus = np.where((np.concatenate([[low_1w[0]], low_1w[:-1]]) - low_1w) > 
                        (high_1w - np.concatenate([[high_1w[0]], high_1w[:-1]])), 
                        np.maximum(np.concatenate([[low_1w[0]], low_1w[:-1]]) - low_1w, 0), 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / (tr_14 + 1e-10)
    di_minus = 100 * dm_minus_14 / (tr_14 + 1e-10)
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_14 = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align weekly ADX to daily
    adx_14_daily = align_htf_to_ltf(prices, df_1w, adx_14)
    
    # Calculate 1d ATR(14) for volatility
    tr1_d = high - low
    tr2_d = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3_d = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_d = np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))
    atr_14 = pd.Series(tr_d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_daily[i]) or np.isnan(lower_20_daily[i]) or 
            np.isnan(adx_14_daily[i]) or np.isnan(atr_14[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Daily price breaks above 1d Donchian upper (20) - bullish breakout
        # 2. Weekly ADX > 25 (strong trend)
        # 3. Volume confirmation: volume > 1.5x average
        if (close[i] > upper_20_daily[i] and
            adx_14_daily[i] > 25 and
            volume_ratio[i] > 1.5):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Daily price breaks below 1d Donchian lower (20) - bearish breakdown
        # 2. Weekly ADX > 25 (strong trend)
        # 3. Volume confirmation: volume > 1.5x average
        elif (close[i] < lower_20_daily[i] and
              adx_14_daily[i] > 25 and
              volume_ratio[i] > 1.5):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_1w_ADX25_Volume_Filter_v1"
timeframe = "1d"
leverage = 1.0