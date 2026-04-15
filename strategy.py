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
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly Supertrend(ATR=10, mult=3) for trend filter
    # ATR calculation
    tr1 = pd.Series(df_1w['high']).rolling(2).apply(lambda x: x[1] - x[0], raw=True)
    tr2 = abs(pd.Series(df_1w['high']) - pd.Series(df_1w['close']).shift(1))
    tr3 = abs(pd.Series(df_1w['low']) - pd.Series(df_1w['close']).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_10 = tr.rolling(window=10, min_periods=10).mean()
    
    # Supertrend calculation
    hl2 = (df_1w['high'] + df_1w['low']) / 2
    upper_band = hl2 + (3 * atr_10)
    lower_band = hl2 - (3 * atr_10)
    
    # Initialize Supertrend
    supertrend = np.full(len(df_1w), np.nan)
    direction = np.full(len(df_1w), np.nan)  # 1 for uptrend, -1 for downtrend
    
    for i in range(10, len(df_1w)):
        if i == 10:
            supertrend[i] = lower_band.iloc[i]
            direction[i] = 1
        else:
            if close.iloc[i-1] > supertrend[i-1]:
                direction[i] = 1
            else:
                direction[i] = -1
            
            if direction[i] == 1:
                supertrend[i] = max(lower_band.iloc[i], supertrend[i-1])
            else:
                supertrend[i] = min(upper_band.iloc[i], supertrend[i-1])
            
            # Reversal conditions
            if direction[i] == 1 and close.iloc[i] < supertrend[i]:
                direction[i] = -1
                supertrend[i] = upper_band.iloc[i]
            elif direction[i] == -1 and close.iloc[i] > supertrend[i]:
                direction[i] = 1
                supertrend[i] = lower_band.iloc[i]
    
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend)
    
    # Calculate weekly EMA(50) for additional trend confirmation
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 12h Donchian channel (20-period) for breakout signals
    donchian_high_20 = pd.Series(close).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(close).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h volume SMA(20) for volume confirmation
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is NaN
        if (np.isnan(supertrend_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or 
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 12h average volume
        vol_confirm = volume[i] > 1.5 * vol_sma_20[i]
        
        # Long conditions:
        # 1. Price breaks above 12h Donchian high (breakout)
        # 2. Price above weekly Supertrend (bullish bias)
        # 3. Price above weekly EMA50 (additional bullish confirmation)
        # 4. Volume confirmation
        if (close[i] > donchian_high_20[i] and 
            close[i] > supertrend_aligned[i] and 
            close[i] > ema_50_1w_aligned[i] and 
            vol_confirm):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below 12h Donchian low (breakdown)
        # 2. Price below weekly Supertrend (bearish bias)
        # 3. Price below weekly EMA50 (additional bearish confirmation)
        # 4. Volume confirmation
        elif (close[i] < donchian_low_20[i] and 
              close[i] < supertrend_aligned[i] and 
              close[i] < ema_50_1w_aligned[i] and 
              vol_confirm):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Supertrend50_EMA50_Donchian20_VolFilter_v1"
timeframe = "12h"
leverage = 1.0