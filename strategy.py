#!/usr/bin/env python3
"""
12h Donchian(20) breakout with 1d ATR filter and volume confirmation.
Trades breakouts only when ATR(10) > 0.8 * ATR(30) (volatility filter).
Long when price breaks above 20-period 12h high + volume > 1.5x 20-period volume MA.
Short when price breaks below 20-period 12h low + volume > 1.5x 20-period volume MA.
Exit on opposite breakout or when ATR(10) < 0.6 * ATR(30) (low volatility).
Designed for 12h to capture strong moves while avoiding chop.
Target: 15-25 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data
    df_12h = get_htf_data(prices, '12h')
    
    # 12h Donchian channels (20-period)
    donch_high = pd.Series(df_12h['high']).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_12h['low']).rolling(window=20, min_periods=20).min().values
    
    # Shift to avoid look-ahead (use completed period's levels)
    donch_high = np.roll(donch_high, 1)
    donch_low = np.roll(donch_low, 1)
    donch_high[0] = np.nan
    donch_low[0] = np.nan
    
    # 1d ATR for volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_30 = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    
    # Align 12h levels and 1d ATR to 12h timeframe (already aligned, but ensure no look-ahead)
    donch_high_12h = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_12h = align_htf_to_ltf(prices, df_12h, donch_low)
    atr_10_12h = align_htf_to_ltf(prices, df_1d, atr_10)
    atr_30_12h = align_htf_to_ltf(prices, df_1d, atr_30)
    
    # Volume confirmation: 20-period volume MA on 12h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # warmup for Donchian and ATR
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high_12h[i]) or np.isnan(donch_low_12h[i]) or
            np.isnan(atr_10_12h[i]) or np.isnan(atr_30_12h[i]) or
            np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        # Volatility filter: only trade when ATR(10) > 0.8 * ATR(30)
        vol_filter = atr_10_12h[i] > 0.8 * atr_30_12h[i]
        
        if position == 0:
            # Long: price breaks above 12h Donchian high with volume spike and vol filter
            if price > donch_high_12h[i] and vol > 1.5 * vol_ma and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian low with volume spike and vol filter
            elif price < donch_low_12h[i] and vol > 1.5 * vol_ma and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below 12h Donchian low OR vol filter fails
            if price < donch_low_12h[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above 12h Donchian high OR vol filter fails
            if price > donch_high_12h[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Volume_ATRFilter"
timeframe = "12h"
leverage = 1.0