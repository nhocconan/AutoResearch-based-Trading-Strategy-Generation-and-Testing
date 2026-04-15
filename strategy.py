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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility regime
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1w Donchian channels (20-period) for structure
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    upper_20_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_20_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align 1d indicators to 12h
    atr_14_12h = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    upper_20_12h = align_htf_to_ltf(prices, df_1w, upper_20_1w)
    lower_20_12h = align_htf_to_ltf(prices, df_1w, lower_20_1w)
    
    # Calculate 12h price position relative to 1w Donchian
    # Normalized position: 0 at lower band, 1 at upper band, 0.5 at midpoint
    donchian_width = upper_20_12h - lower_20_12h
    donchian_position = np.where(
        donchian_width > 0,
        (close - lower_20_12h) / donchian_width,
        0.5
    )
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Precompute session filter (00-24 UTC for 12h - less restrictive)
    hours = prices.index.hour
    in_session = (hours >= 0) & (hours <= 23)  # Always true for 12h, kept for structure
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_12h[i]) or np.isnan(ema_50_12h[i]) or 
            np.isnan(upper_20_12h[i]) or np.isnan(lower_20_12h[i]) or 
            np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Regime filter: use 1d ATR normalized by price to detect volatility expansion
        atr_ratio = atr_14_12h[i] / close[i]
        vol_expansion = atr_ratio > 0.015  # 1.5% ATR threshold for volatile regimes
        
        # Trend filter: price relative to 1d EMA50
        price_vs_ema = close[i] / ema_50_12h[i]
        
        # Long conditions:
        # 1. Price in upper half of 1w Donchian channel (bullish structure)
        # 2. Volatility expansion (avoid low volatility chop)
        # 3. Price above 1d EMA50 (bullish trend bias)
        # 4. Volume confirmation: volume > 1.2x average
        if (donchian_position[i] > 0.5 and
            vol_expansion and
            price_vs_ema > 1.0 and
            volume_ratio[i] > 1.2):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price in lower half of 1w Donchian channel (bearish structure)
        # 2. Volatility expansion (avoid low volatility chop)
        # 3. Price below 1d EMA50 (bearish trend bias)
        # 4. Volume confirmation: volume > 1.2x average
        elif (donchian_position[i] < 0.5 and
              vol_expansion and
              price_vs_ema < 1.0 and
              volume_ratio[i] > 1.2):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_1d_ATR_EMA_1w_Donchian_Position_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0