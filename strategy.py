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
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    highest_20 = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align to 1d timeframe (wait for weekly bar to close)
    highest_20_aligned = align_htf_to_ltf(prices, df_1w, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1w, lowest_20)
    
    # Calculate weekly ATR(14) for volatility filter
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = np.abs(df_1w['high'] - np.concatenate([[df_1w['close'].iloc[0]], df_1w['close'].iloc[:-1]]))
    tr3 = np.abs(df_1w['low'] - np.concatenate([[df_1w['close'].iloc[0]], df_1w['close'].iloc[:-1]]))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1w = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or 
            np.isnan(atr_14_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when weekly ATR is elevated (> 0.5% of price)
        vol_filter = atr_14_1w_aligned[i] > 0.005 * close[i]
        
        # Long conditions:
        # 1. Price breaks above weekly Donchian high (20-period breakout)
        # 2. Volatility filter
        if (close[i] > highest_20_aligned[i] and vol_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below weekly Donchian low (20-period breakout)
        # 2. Volatility filter
        elif (close[i] < lowest_20_aligned[i] and vol_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyDonchian20_VolFilter_v1"
timeframe = "1d"
leverage = 1.0