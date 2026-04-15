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
    
    # Calculate weekly Donchian(20) channels for breakout detection
    donchian_high_20 = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    
    # Calculate weekly ATR(14) for volatility filter
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = np.abs(df_1w['high'] - np.concatenate([[df_1w['close'].iloc[0]], df_1w['close'].iloc[:-1]]))
    tr3 = np.abs(df_1w['low'] - np.concatenate([[df_1w['close'].iloc[0]], df_1w['close'].iloc[:-1]]))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1w = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(atr_14_1w_aligned[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when weekly ATR is elevated (> 0.3% of price)
        vol_filter = atr_14_1w_aligned[i] > 0.003 * close[i]
        
        # Long conditions:
        # 1. Price breaks above weekly Donchian(20) high
        # 2. Volume confirmation: volume > 2.0x average
        # 3. Volatility filter
        if (close[i] > donchian_high_20_aligned[i] and
            volume_ratio[i] > 2.0 and
            vol_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below weekly Donchian(20) low
        # 2. Volume confirmation: volume > 2.0x average
        # 3. Volatility filter
        elif (close[i] < donchian_low_20_aligned[i] and
              volume_ratio[i] > 2.0 and
              vol_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Weekly_Donchian20_Volume_Breakout_v1"
timeframe = "12h"
leverage = 1.0