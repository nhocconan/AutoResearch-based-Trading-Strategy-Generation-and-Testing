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
    
    # Get weekly HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly EMA(21) for trend filter
    ema_21_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate daily Donchian(20) channels for entry timing
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    donchian_high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(donchian_high_20_aligned[i]) or 
            np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: price relative to weekly EMA21
        weekly_bullish = close[i] > ema_21_1w_aligned[i]
        weekly_bearish = close[i] < ema_21_1w_aligned[i]
        
        # Long conditions:
        # 1. Weekly bullish trend (price above weekly EMA21)
        # 2. Price breaks above daily Donchian(20) high with volume confirmation
        if (weekly_bullish and
            close[i] > donchian_high_20_aligned[i] and
            volume_ratio[i] > 2.0):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Weekly bearish trend (price below weekly EMA21)
        # 2. Price breaks below daily Donchian(20) low with volume confirmation
        elif (weekly_bearish and
              close[i] < donchian_low_20_aligned[i] and
              volume_ratio[i] > 2.0):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WeeklyEMA21_Donchian20_Volume_Breakout_v1"
timeframe = "6h"
leverage = 1.0