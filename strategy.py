#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeATRFilter_V1
Hypothesis: Donchian(20) breakout with volume confirmation (>1.5x 20-bar MA) and ATR-based stoploss works on 4h timeframe for BTC and ETH in both bull and bear markets. Uses 1d timeframe for trend filter (EMA34). Target: 19-50 trades/year per symbol (75-200 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for EMA34 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Donchian(20) channels on 4h
    high = prices['high'].values
    low = prices['low'].values
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period average
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
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation (>1.5x average to reduce trades)
        volume_ok = volume > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and above 1d EMA34
            if price > donchian_high[i] and price > ema_34_aligned[i]:
                if volume_ok:
                    signals[i] = 0.25
                    position = 1
            # Short: price breaks below Donchian low with volume and below 1d EMA34
            elif price < donchian_low[i] and price < ema_34_aligned[i]:
                if volume_ok:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: price closes below Donchian low or ATR stoploss
            if price < donchian_low[i] or price < prices['close'].iloc[i-1] - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above Donchian high or ATR stoploss
            if price > donchian_high[i] or price > prices['close'].iloc[i-1] + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_VolumeATRFilter_V1"
timeframe = "4h"
leverage = 1.0