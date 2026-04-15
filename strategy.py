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
    
    # Get daily data for context
    daily = get_htf_data(prices, '1d')
    daily_high = daily['high'].values
    daily_low = daily['low'].values
    daily_close = daily['close'].values
    
    # Calculate daily ADX for trend strength
    # True Range
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr3 = np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = daily_high - np.concatenate([[daily_high[0]], daily_high[:-1]])
    down_move = np.concatenate([[daily_low[0]], daily_low[:-1]]) - daily_low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm14 / (tr14 + 1e-10)
    minus_di = 100 * minus_dm14 / (tr14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align daily ADX to 6h timeframe
    adx_6h = align_htf_to_ltf(prices, daily, adx)
    
    # Calculate daily EMA for trend direction
    ema_20 = pd.Series(daily_close).ewm(span=20, adjust=False).mean().values
    ema_50 = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    
    # Align EMAs to 6h timeframe
    ema_20_6h = align_htf_to_ltf(prices, daily, ema_20)
    ema_50_6h = align_htf_to_ltf(prices, daily, ema_50)
    
    # Calculate daily volume ratio for volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_6h[i]) or np.isnan(ema_20_6h[i]) or np.isnan(ema_50_6h[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Trend strength filter: ADX > 25
        if adx_6h[i] < 25:
            signals[i] = 0.0
            continue
        
        # Trend direction: EMA20 > EMA50 = uptrend, EMA20 < EMA50 = downtrend
        # Volume filter: volume > 1.5x average
        if (ema_20_6h[i] > ema_50_6h[i] and vol_ratio[i] > 1.5):
            # Uptrend with volume - go long
            signals[i] = 0.25
        elif (ema_20_6h[i] < ema_50_6h[i] and vol_ratio[i] > 1.5):
            # Downtrend with volume - go short
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_ADX_EMA_Volume_Trend"
timeframe = "6h"
leverage = 1.0