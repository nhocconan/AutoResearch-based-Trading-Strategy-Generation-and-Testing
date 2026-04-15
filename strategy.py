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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Donchian(20) channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # Calculate 6h ADX(14) for trend strength filter
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Directional Movement
    up_move = high - np.concatenate([[high[0]], high[:-1]])
    down_move = np.concatenate([[low[0]], low[:-1]]) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    # Smoothed values
    tr_ma = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_ma = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_ma = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    # DI and DX
    plus_di = 100 * plus_dm_ma / (tr_ma + 1e-10)
    minus_di = 100 * minus_dm_ma / (tr_ma + 1e-10)
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high_20_aligned[i]) or 
            np.isnan(donchian_low_20_aligned[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade when price is above/below 1d EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Trend strength filter: only trade when ADX > 25
        strong_trend = adx[i] > 25
        
        # Long conditions:
        # 1. Price above 1d EMA50 (bullish bias)
        # 2. Price breaks above 1d Donchian(20) high
        # 3. Strong trend (ADX > 25)
        if (price_above_ema and 
            close[i] > donchian_high_20_aligned[i] and
            strong_trend):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below 1d EMA50 (bearish bias)
        # 2. Price breaks below 1d Donchian(20) low
        # 3. Strong trend (ADX > 25)
        elif (price_below_ema and 
              close[i] < donchian_low_20_aligned[i] and
              strong_trend):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_EMA50_Donchian20_ADX_Filter_v1"
timeframe = "6h"
leverage = 1.0