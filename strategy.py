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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Donchian channel (20-period) for breakout signals
    highest_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    highest_20_aligned = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_20)
    
    # Calculate daily ATR(14) for volatility filter and position sizing
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily volume average (20-period) for volume confirmation
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current daily volume above 20-period average
        vol_filter = df_1d['volume'].iloc[i] > vol_ma_20_aligned[i] if i < len(df_1d) else False
        
        # Long breakout: price breaks above 20-day high with volume confirmation
        if (close[i] > highest_20_aligned[i] and vol_filter):
            signals[i] = 0.25
            
        # Short breakout: price breaks below 20-day low with volume confirmation
        elif (close[i] < lowest_20_aligned[i] and vol_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian20_1d_Volume_Breakout_v1"
timeframe = "4h"
leverage = 1.0