#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_VolumeTrend_Regime_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ATR for volatility regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr_1d = np.maximum(high_1d - low_1d, np.absolute(high_1d - np.roll(close_1d, 1)), np.absolute(low_1d - np.roll(close_1d, 1)))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_avg = pd.Series(atr_1d).rolling(window=30, min_periods=30).mean().values
    atr_ratio = atr_1d / atr_1d_avg
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # 4h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    
    # 4h volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 30, 30)  # Donchian + volume + ATR ratio
    
    for i in range(start_idx, n):
        if np.isnan(atr_ratio_aligned[i]) or np.isnan(donchian_upper[i]) or \
           np.isnan(donchian_lower[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr_ratio_val = atr_ratio_aligned[i]
        
        # Regime filter: high volatility (ATR ratio > 1.2) = trend following
        # Low volatility (ATR ratio < 0.8) = range bound (avoid)
        vol_regime_ok = atr_ratio_val > 1.2
        
        # Volume confirmation
        volume_ok = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: breakout above Donchian upper + volume + high vol regime
            if price > donchian_upper[i-1] and volume_ok and vol_regime_ok:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian lower + volume + high vol regime
            elif price < donchian_lower[i-1] and volume_ok and vol_regime_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below Donchian middle or volatility drops
            donchian_mid = (donchian_upper[i-1] + donchian_lower[i-1]) / 2
            if price < donchian_mid or atr_ratio_val < 0.9:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above Donchian middle or volatility drops
            donchian_mid = (donchian_upper[i-1] + donchian_lower[i-1]) / 2
            if price > donchian_mid or atr_ratio_val < 0.9:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals