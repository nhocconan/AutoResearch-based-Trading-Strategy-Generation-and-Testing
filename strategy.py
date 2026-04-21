#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeATRFilter_Regime_V1
Hypothesis: Donchian(20) breakouts with volume confirmation, ATR-based stoploss, and choppiness regime filter work on 4h for BTC and ETH in both bull and bear markets. Uses 12h EMA50 for trend filter and 12h choppiness index to avoid ranging markets. Target: 20-50 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data once for trend filter and choppiness
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 12h Choppiness Index (CHOP) for regime filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_arr = df_12h['close'].values
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h_arr, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h_arr, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_12h.sum() / (highest_high - lowest_low)) / np.log10(14)
    chop = np.where((highest_high - lowest_low) == 0, 50, chop)  # avoid div/0
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # ATR for stoploss (4h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation
        volume_ok = volume > 1.5 * vol_ma[i]
        
        # 12h trend filter
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Regime filter: avoid choppy markets (CHOP > 61.8 = range)
        trending_market = chop_aligned[i] < 61.8
        
        if position == 0:
            # Long: price breaks above Donchian high in uptrend with volume and trending regime
            if uptrend and volume_ok and trending_market:
                if price > high_max[i]:
                    signals[i] = 0.30
                    position = 1
            # Short: price breaks below Donchian low in downtrend with volume and trending regime
            elif downtrend and volume_ok and trending_market:
                if price < low_min[i]:
                    signals[i] = -0.30
                    position = -1
        
        elif position == 1:
            # Exit: price reaches Donchian low or stoploss
            if price <= low_min[i] or price < prices['close'].iloc[i-1] - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit: price reaches Donchian high or stoploss
            if price >= high_max[i] or price > prices['close'].iloc[i-1] + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_VolumeATRFilter_Regime_V1"
timeframe = "4h"
leverage = 1.0