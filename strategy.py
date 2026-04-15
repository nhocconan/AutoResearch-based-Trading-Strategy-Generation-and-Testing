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
    
    # Get 1w HTF data once before loop (HTF=1w per instructions)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(21) for HTF trend filter
    ema_21_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate weekly Donchian(20) channels
    donchian_high_20_1w = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low_20_1w = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_20_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20_1w)
    donchian_low_20_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20_1w)
    
    # Calculate 1d volume ratio (current vs 20-period average from 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_ratio = volume / (vol_ma_20_1d_aligned + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(donchian_high_20_1w_aligned[i]) or 
            np.isnan(donchian_low_20_1w_aligned[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Price above weekly EMA21 (bullish HTF bias)
        # 2. Price breaks above weekly Donchian(20) high with volume confirmation
        # 3. Volume > 1.5x average
        if (close[i] > ema_21_1w_aligned[i] and
            close[i] > donchian_high_20_1w_aligned[i] and
            volume_ratio[i] > 1.5):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below weekly EMA21 (bearish HTF bias)
        # 2. Price breaks below weekly Donchian(20) low with volume confirmation
        # 3. Volume > 1.5x average
        elif (close[i] < ema_21_1w_aligned[i] and
              close[i] < donchian_low_20_1w_aligned[i] and
              volume_ratio[i] > 1.5):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyEMA21_Donchian20_Volume_Breakout_v1"
timeframe = "1d"
leverage = 1.0