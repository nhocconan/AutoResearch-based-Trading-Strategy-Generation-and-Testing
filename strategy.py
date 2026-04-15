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
    donchian_high_20 = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # Calculate daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily volume SMA(20) for volume confirmation
    vol_sma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 0.75x daily average volume
        # Approximate daily volume from 6h: use 4x current 6h volume as proxy for daily
        vol_confirm = volume[i] > 0.75 * vol_sma_20_aligned[i]  # Conservative threshold
        
        # Long conditions:
        # 1. Price breaks above daily Donchian high (breakout)
        # 2. Price above daily EMA34 (bullish bias)
        # 3. Volume confirmation
        if (close[i] > donchian_high_20_aligned[i] and 
            close[i] > ema_34_1d_aligned[i] and 
            vol_confirm):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below daily Donchian low (breakdown)
        # 2. Price below daily EMA34 (bearish bias)
        # 3. Volume confirmation
        elif (close[i] < donchian_low_20_aligned[i] and 
              close[i] < ema_34_1d_aligned[i] and 
              vol_confirm):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Donchian20_EMA34_VolFilter_v1"
timeframe = "6h"
leverage = 1.0