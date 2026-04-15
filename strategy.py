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
    
    # Calculate 12h EMA(34) for trend filter
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 12h ATR(14) for volatility filter
    tr1 = pd.Series(df_12h['high'].values - df_12h['low'].values)
    tr2 = pd.Series(np.abs(df_12h['high'].values - np.roll(df_12h['close'].values, 1)))
    tr3 = pd.Series(np.abs(df_12h['low'].values - np.roll(df_12h['close'].values, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    
    # Calculate 12h volume SMA(20) for volume confirmation
    vol_sma_20_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_sma_20_12h)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(atr_14_12h_aligned[i]) or 
            np.isnan(vol_sma_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.2x 12h average volume
        vol_confirm = volume[i] > 1.2 * vol_sma_20_12h_aligned[i]
        
        # Volatility filter: only trade when 12h ATR > its 50-period SMA (avoid low vol choppy periods)
        atr_sma_50_12h = pd.Series(atr_14_12h_aligned).rolling(window=50, min_periods=50).mean().values
        vol_filter = not np.isnan(atr_sma_50_12h[i]) and atr_14_12h_aligned[i] > atr_sma_50_12h[i]
        
        # Long conditions:
        # 1. Price above 12h EMA34 (bullish bias)
        # 2. Volume confirmation
        # 3. Volatility filter (trade only in sufficient volatility)
        if (close[i] > ema_34_12h_aligned[i] and 
            vol_confirm and 
            vol_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below 12h EMA34 (bearish bias)
        # 2. Volume confirmation
        # 3. Volatility filter
        elif (close[i] < ema_34_12h_aligned[i] and 
              vol_confirm and 
              vol_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_EMA34_VolVolFilter_v1"
timeframe = "6h"
leverage = 1.0