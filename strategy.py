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
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h pivot points (R1, S1, R2, S2)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    pivot = (high_12h + low_12h + close_12h) / 3.0
    r1 = 2 * pivot - low_12h
    s1 = 2 * pivot - high_12h
    r2 = pivot + (high_12h - low_12h)
    s2 = pivot - (high_12h - low_12h)
    
    # Align 12h pivots to 6h
    r1_6h = align_htf_to_ltf(prices, df_12h, r1)
    s1_6h = align_htf_to_ltf(prices, df_12h, s1)
    r2_6h = align_htf_to_ltf(prices, df_12h, r2)
    s2_6h = align_htf_to_ltf(prices, df_12h, s2)
    
    # Get 1d HTF data for trend filter (ADX)
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
    
    # Directional Movement
    up_move = high_1d - np.concatenate([[high_1d[0]], high_1d[:-1]])
    down_move = np.concatenate([[low_1d[0]], low_1d[:-1]]) - low_1d
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI values
    plus_di_14 = 100 * plus_dm_14 / (tr_14 + 1e-10)
    minus_di_14 = 100 * minus_dm_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_14 = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 6h
    adx_1d_6h = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate 6h ATR(14) for volatility filter
    tr1_6h = high - low
    tr2_6h = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3_6h = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr_14_6h = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio_6h = volume / (vol_ma_20_6h + 1e-10)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or
            np.isnan(adx_1d_6h[i]) or np.isnan(atr_14_6h[i]) or np.isnan(volume_ratio_6h[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 6h price breaks above 12h R2 (strong bullish breakout)
        # 2. 1d ADX > 25 (trending market)
        # 3. Volume confirmation: volume > 1.3x average
        # 4. Volatility filter: ATR > 0.4% of price (avoid low volatility chop)
        if (close[i] > r2_6h[i] and
            adx_1d_6h[i] > 25.0 and
            volume_ratio_6h[i] > 1.3 and
            atr_14_6h[i] > 0.004 * close[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. 6h price breaks below 12h S2 (strong bearish breakout)
        # 2. 1d ADX > 25 (trending market)
        # 3. Volume confirmation: volume > 1.3x average
        # 4. Volatility filter: ATR > 0.4% of price
        elif (close[i] < s2_6h[i] and
              adx_1d_6h[i] > 25.0 and
              volume_ratio_6h[i] > 1.3 and
              atr_14_6h[i] > 0.004 * close[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_12h_PivotR2S2_Breakout_ADX25_Volume_Filter"
timeframe = "6h"
leverage = 1.0