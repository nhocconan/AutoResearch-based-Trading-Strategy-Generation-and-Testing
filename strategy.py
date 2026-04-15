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
    
    # Get 1d HTF data once before loop (primary HTF for 6h strategy)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # +DM and -DM
    up_move = high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])
    down_move = np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed +DM, -DM, TR
    tr_period = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm_period = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_period = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_period / (tr_period + 1e-10)
    minus_di = 100 * minus_dm_period / (tr_period + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align 1d ADX to 6h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1d Donchian channels (20-period) for breakout levels
    upper_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian to 6h
    upper_20_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_20_1d)
    lower_20_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_20_1d)
    
    # Calculate 6h ATR(14) for volatility filter
    tr1_6h = high - low
    tr2_6h = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3_6h = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr_6h = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Pre-compute session filter (08-20 UTC) - using DatetimeIndex directly
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(upper_20_1d_aligned[i]) or 
            np.isnan(lower_20_1d_aligned[i]) or np.isnan(atr_6h[i]) or 
            np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 6h price breaks above 1d Donchian upper (20)
        # 2. 1d ADX > 25 (strong trend)
        # 3. Volume confirmation: volume > 1.3x average
        # 4. Volatility filter: ATR > 0.4% of price (avoid low volatility chop)
        if (close[i] > upper_20_1d_aligned[i] and
            adx_1d_aligned[i] > 25 and
            volume_ratio[i] > 1.3 and
            atr_6h[i] > 0.004 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 6h price breaks below 1d Donchian lower (20)
        # 2. 1d ADX > 25 (strong trend)
        # 3. Volume confirmation: volume > 1.3x average
        # 4. Volatility filter: ATR > 0.4% of price
        elif (close[i] < lower_20_1d_aligned[i] and
              adx_1d_aligned[i] > 25 and
              volume_ratio[i] > 1.3 and
              atr_6h[i] > 0.004 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_ADX25_Donchian20_1d_Breakout_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0