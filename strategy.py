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
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Williams %R (14-period)
    highest_high_14 = pd.Series(df_1w['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1w['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - df_1w['close'].values) / (highest_high_14 - lowest_low_14 + 1e-10)
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r, additional_delay_bars=0)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily ADX (14-period)
    plus_dm = np.where((df_1d['high'].values - np.roll(df_1d['high'].values, 1)) > 
                       (np.roll(df_1d['low'].values, 1) - df_1d['low'].values), 
                       np.maximum(df_1d['high'].values - np.roll(df_1d['high'].values, 1), 0), 0)
    minus_dm = np.where((np.roll(df_1d['low'].values, 1) - df_1d['low'].values) > 
                        (df_1d['high'].values - np.roll(df_1d['high'].values, 1)), 
                        np.maximum(np.roll(df_1d['low'].values, 1) - df_1d['low'].values, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_14_1d + 1e-10)
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_14_1d + 1e-10)
    dx_14 = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_14_1d = pd.Series(dx_14).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_14_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14_1d)
    
    # Calculate 6h Donchian(20) channels
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(adx_14_1d_aligned[i]) or 
            np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Weekly Williams %R oversold (< -80) - potential reversal up
        # 2. Daily ADX > 25 - trending market
        # 3. Price breaks above 6h Donchian(20) high - breakout confirmation
        if (williams_r_aligned[i] < -80 and
            adx_14_1d_aligned[i] > 25 and
            close[i] > donchian_high_20[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Weekly Williams %R overbought (> -20) - potential reversal down
        # 2. Daily ADX > 25 - trending market
        # 3. Price breaks below 6h Donchian(20) low - breakdown confirmation
        elif (williams_r_aligned[i] > -20 and
              adx_14_1d_aligned[i] > 25 and
              close[i] < donchian_low_20[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR_ADX_Donchian20_Breakout_v1"
timeframe = "6h"
leverage = 1.0