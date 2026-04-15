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
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA(21) for trend
    ema_21_4h = pd.Series(df_4h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Get 1d HTF data for regime filter (choppiness)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for choppiness
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d true range sum and high-low range for choppiness
    atr_sum_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = np.log10(hh_14 - ll_14 + 1e-10) * np.sqrt(14)
    chop_1d = 100 * np.log10(atr_sum_14 / chop_denom) / np.log10(10)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 1h Donchian(20) for entry timing
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1h volume ratio
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Regime: choppiness > 61.8 = range, < 38.2 = trending
        is_ranging = chop_1d_aligned[i] > 61.8
        is_trending = chop_1d_aligned[i] < 38.2
        
        # Long conditions:
        # 1. 4h EMA21 uptrend (price above EMA)
        # 2. 1h price breaks above Donchian(20) upper
        # 3. Volume confirmation: volume > 1.5x average
        # 4. In ranging OR trending regime (both allowed)
        if (close[i] > ema_21_4h_aligned[i] and
            close[i] > highest_20[i] and
            volume_ratio[i] > 1.5):
            signals[i] = 0.20
            
        # Short conditions:
        # 1. 4h EMA21 downtrend (price below EMA)
        # 2. 1h price breaks below Donchian(20) lower
        # 3. Volume confirmation: volume > 1.5x average
        elif (close[i] < ema_21_4h_aligned[i] and
              close[i] < lowest_20[i] and
              volume_ratio[i] > 1.5):
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_4h_EMA21_1d_Chop_Donchian20_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0