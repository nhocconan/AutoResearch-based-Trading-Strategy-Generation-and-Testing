#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeFilter_V1
Hypothesis: 4h Donchian(20) breakouts in the direction of 1d EMA50 trend with volume confirmation work for BTC and ETH in both bull and bear markets. Uses ATR-based stoploss. Target: 20-50 trades/year per symbol (80-200 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h Donchian(20) breakout levels
    high = prices['high'].values
    low = prices['low'].values
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period average (approx 6.7h on 4h)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(prices['close'].values, 1))
    tr3 = np.abs(low - np.roll(prices['close'].values, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation
        volume_ok = volume > 1.5 * vol_ma[i]
        
        # 1d trend filter
        uptrend = price > ema_50_1d_aligned[i]
        downtrend = price < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high in uptrend with volume
            if uptrend and volume_ok:
                if price > donchian_high[i]:
                    signals[i] = 0.30
                    position = 1
            # Short: price breaks below Donchian low in downtrend with volume
            elif downtrend and volume_ok:
                if price < donchian_low[i]:
                    signals[i] = -0.30
                    position = -1
        
        elif position == 1:
            # Exit: price reaches Donchian low or stoploss
            if price <= donchian_low[i] or price < prices['close'].iloc[i-1] - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit: price reaches Donchian high or stoploss
            if price >= donchian_high[i] or price > prices['close'].iloc[i-1] + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeFilter_V1"
timeframe = "4h"
leverage = 1.0