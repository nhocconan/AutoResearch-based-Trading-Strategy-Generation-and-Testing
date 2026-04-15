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
    open_time = prices['open_time']
    
    # Pre-compute hour filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Supertrend (ATR=10, mult=3)
    tr1 = df_4h['high'] - df_4h['low']
    tr2 = np.abs(df_4h['high'] - np.concatenate([[df_4h['close'].iloc[0]], df_4h['close'].iloc[:-1]]))
    tr3 = np.abs(df_4h['low'] - np.concatenate([[df_4h['close'].iloc[0]], df_4h['close'].iloc[:-1]]))
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_4h = pd.Series(tr_4h).ewm(span=10, adjust=False, min_periods=10).mean().values
    hl2_4h = (df_4h['high'].values + df_4h['low'].values) / 2
    upper_band_4h = hl2_4h + 3 * atr_4h
    lower_band_4h = hl2_4h - 3 * atr_4h
    supertrend_4h = np.full_like(close, np.nan)
    for i in range(len(df_4h)):
        if i == 0:
            supertrend_4h[i] = upper_band_4h[i]
        else:
            if supertrend_4h[i-1] == upper_band_4h[i-1]:
                supertrend_4h[i] = upper_band_4h[i] if close_4h[i] <= upper_band_4h[i] else lower_band_4h[i]
            else:
                supertrend_4h[i] = lower_band_4h[i] if close_4h[i] >= lower_band_4h[i] else upper_band_4h[i]
    close_4h = df_4h['close'].values
    supertrend_4h_aligned = align_htf_to_ltf(prices, df_4h, supertrend_4h)
    
    # Calculate 1d Donchian(20) channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    donchian_high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(supertrend_4h_aligned[i]) or 
            np.isnan(donchian_high_20_aligned[i]) or 
            np.isnan(donchian_low_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. 4h Supertrend bullish (price above supertrend)
        # 2. Price breaks above 1d Donchian(20) high
        if (close[i] > supertrend_4h_aligned[i] and
            close[i] > donchian_high_20_aligned[i]):
            signals[i] = 0.20
            
        # Short conditions:
        # 1. 4h Supertrend bearish (price below supertrend)
        # 2. Price breaks below 1d Donchian(20) low
        elif (close[i] < supertrend_4h_aligned[i] and
              close[i] < donchian_low_20_aligned[i]):
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_Supertrend4h_Donchian1d_08-20UTC_v1"
timeframe = "1h"
leverage = 1.0