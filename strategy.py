#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily ATR(14)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 4h Donchian(20) channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    donchian_high_20_4h = pd.Series(df_4h['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low_20_4h = pd.Series(df_4h['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_20_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_20_4h)
    donchian_low_20_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_20_4h)
    
    # Calculate 4h volume SMA(20) for volume confirmation
    volume_sma_20_4h = pd.Series(df_4h['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_sma_20_4h)
    
    signals = np.zeros(n)
    position = 0  # track current position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_20_4h_aligned[i]) or np.isnan(donchian_low_20_4h_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(volume_sma_20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when daily ATR is elevated (> 0.3% of price)
        vol_filter = atr_14_1d_aligned[i] > 0.003 * close[i]
        
        # Volume confirmation: current 4h volume > 20-period average
        vol_confirm = volume[i] > volume_sma_20_4h_aligned[i]
        
        # Long conditions:
        # 1. Price breaks above 4h Donchian(20) high
        # 2. Volatility filter
        # 3. Volume confirmation
        if (close[i] > donchian_high_20_4h_aligned[i] and
            vol_filter and
            vol_confirm):
            signals[i] = 0.25
            position = 1
            
        # Short conditions:
        # 1. Price breaks below 4h Donchian(20) low
        # 2. Volatility filter
        # 3. Volume confirmation
        elif (close[i] < donchian_low_20_4h_aligned[i] and
              vol_filter and
              vol_confirm):
            signals[i] = -0.25
            position = -1
            
        # Exit conditions: close position when price crosses opposite Donchian band
        elif position == 1 and close[i] < donchian_low_20_4h_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > donchian_high_20_4h_aligned[i]:
            signals[i] = 0.0
            position = 0
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_Vol_ATR_Filter_v1"
timeframe = "4h"
leverage = 1.0