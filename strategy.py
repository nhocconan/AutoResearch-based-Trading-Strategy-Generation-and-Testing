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
    
    # Get weekly HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly Donchian channel (20-period) for breakout signals
    donchian_high_20 = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    
    # Calculate weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate weekly volume SMA(20) for volume confirmation
    vol_sma_20 = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_aligned = align_htf_to_ltf(prices, df_1w, vol_sma_20)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current daily volume > 1.5x weekly average volume
        # Approximate weekly volume from daily: use 5x current daily volume as proxy for weekly
        vol_confirm = volume[i] > 0.3 * vol_sma_20_aligned[i]  # Conservative threshold
        
        # Long conditions:
        # 1. Price breaks above weekly Donchian high (breakout)
        # 2. Price above weekly EMA34 (bullish bias)
        # 3. Volume confirmation
        if (close[i] > donchian_high_20_aligned[i] and 
            close[i] > ema_34_1w_aligned[i] and 
            vol_confirm):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below weekly Donchian low (breakdown)
        # 2. Price below weekly EMA34 (bearish bias)
        # 3. Volume confirmation
        elif (close[i] < donchian_low_20_aligned[i] and 
              close[i] < ema_34_1w_aligned[i] and 
              vol_confirm):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_EMA34_VolFilter_1w_v1"
timeframe = "1d"
leverage = 1.0