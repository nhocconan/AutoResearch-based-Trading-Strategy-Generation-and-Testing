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
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h ATR(14) for volatility regime filter
    tr1 = df_12h['high'] - df_12h['low']
    tr2 = np.abs(df_12h['high'] - np.concatenate([[df_12h['close'].iloc[0]], df_12h['close'].iloc[:-1]]))
    tr3 = np.abs(df_12h['low'] - np.concatenate([[df_12h['close'].iloc[0]], df_12h['close'].iloc[:-1]]))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    
    # Calculate 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 6h Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_12h_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when 12h ATR is elevated (> 0.8% of price)
        vol_filter = atr_14_12h_aligned[i] > 0.008 * close[i]
        
        # Long conditions:
        # 1. Price breaks above 6h Donchian upper channel (breakout)
        # 2. Price above 12h EMA50 (bullish bias from higher timeframe)
        # 3. Volatility filter ensures we trade during active regimes
        if (close[i] > highest_20[i] and
            close[i] > ema_50_12h_aligned[i] and
            vol_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below 6h Donchian lower channel (breakdown)
        # 2. Price below 12h EMA50 (bearish bias from higher timeframe)
        # 3. Volatility filter ensures we trade during active regimes
        elif (close[i] < lowest_20[i] and
              close[i] < ema_50_12h_aligned[i] and
              vol_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Donchian20_EMA50_12h_VolFilter_v1"
timeframe = "6h"
leverage = 1.0