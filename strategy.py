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
    close_1d = df_1d['close']
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Williams %R(14) for mean reversion signals
    highest_high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14 + 1e-10)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 6h ADX(14) for regime filter
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_14 + 1e-10)
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_14 + 1e-10)
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when ADX > 25 (trending market)
        trending = adx[i] > 25
        
        # Long conditions:
        # 1. Daily EMA(50) uptrend (price above EMA50)
        # 2. Williams %R oversold (< -80) - mean reentry opportunity
        # 3. Trending regime filter (ADX > 25)
        if (close[i] > ema_50_1d_aligned[i] and
            williams_r_aligned[i] < -80 and
            trending):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Daily EMA(50) downtrend (price below EMA50)
        # 2. Williams %R overbought (> -20) - mean reentry opportunity
        # 3. Trending regime filter (ADX > 25)
        elif (close[i] < ema_50_1d_aligned[i] and
              williams_r_aligned[i] > -20 and
              trending):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_EMA50_WilliamsR_ADX_Trend_MeanReversion_v1"
timeframe = "6h"
leverage = 1.0