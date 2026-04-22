#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Daily data for ATR-based volatility regime
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 14-period ATR on daily
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14d_avg = pd.Series(atr_14d).rolling(window=30, min_periods=30).mean().values
    
    # Volatility regime: high volatility when current ATR > 1.5 * 30-day average ATR
    vol_regime_high = atr_14d > 1.5 * atr_14d_avg
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 20-period Donchian channels on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume surge: current volume > 2.0 * 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high in high volatility regime with volume surge
            if close[i] > donchian_high[i] and vol_regime_high[i] and vol_surge[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low in high volatility regime with volume surge
            elif close[i] < donchian_low[i] and vol_regime_high[i] and vol_surge[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to midpoint of Donchian channel
            midpoint = (donchian_high[i] + donchian_low[i]) / 2.0
            if position == 1:
                if close[i] < midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_VolRegime_VolSurge_v1"
timeframe = "4h"
leverage = 1.0