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
    
    # Calculate 12h Donchian channel (20-period) for breakout signals
    donchian_high_20 = pd.Series(df_12h['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(df_12h['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_20)
    
    # Calculate 12h EMA(34) for trend filter
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 12h volume SMA(20) for volume confirmation
    vol_sma_20 = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_sma_20)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 12h average volume
        vol_confirm = volume[i] > 1.5 * vol_sma_20_aligned[i]
        
        # Long conditions:
        # 1. Price breaks above 12h Donchian high (breakout)
        # 2. Price above 12h EMA34 (bullish bias)
        # 3. Volume confirmation
        if (close[i] > donchian_high_20_aligned[i] and 
            close[i] > ema_34_12h_aligned[i] and 
            vol_confirm):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below 12h Donchian low (breakdown)
        # 2. Price below 12h EMA34 (bearish bias)
        # 3. Volume confirmation
        elif (close[i] < donchian_low_20_aligned[i] and 
              close[i] < ema_34_12h_aligned[i] and 
              vol_confirm):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian20_EMA34_VolFilter_v1"
timeframe = "12h"
leverage = 1.0