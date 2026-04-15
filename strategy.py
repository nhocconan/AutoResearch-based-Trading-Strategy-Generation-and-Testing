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
    
    # Get 12h HTF data once before loop (primary=4h, HTF=12h)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period) for structure
    highest_20_12h = pd.Series(df_12h['high']).rolling(window=20, min_periods=20).max().values
    lowest_20_12h = pd.Series(df_12h['low']).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA(21) for trend filter
    ema_21_12h = pd.Series(df_12h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align HTF indicators to 4h timeframe
    highest_20_12h_aligned = align_htf_to_ltf(prices, df_12h, highest_20_12h)
    lowest_20_12h_aligned = align_htf_to_ltf(prices, df_12h, lowest_20_12h)
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20_12h_aligned[i]) or np.isnan(lowest_20_12h_aligned[i]) or 
            np.isnan(ema_21_12h_aligned[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 12h Donchian breakout/breakdown
        # 2. 12h EMA21 trend filter
        # 3. 4h volume confirmation: volume > 1.8x average
        # 4. Discrete position sizing: 0.25
        
        # Long conditions: 12h Donchian breakout above + 12h uptrend + volume confirmation
        if (close[i] > highest_20_12h_aligned[i] and      # 12h Donchian breakout
            close[i] > ema_21_12h_aligned[i] and         # 12h uptrend filter
            volume_ratio[i] > 1.8):                      # Volume confirmation
            signals[i] = 0.25
            
        # Short conditions: 12h Donchian breakdown below + 12h downtrend + volume confirmation
        elif (close[i] < lowest_20_12h_aligned[i] and    # 12h Donchian breakdown
              close[i] < ema_21_12h_aligned[i] and       # 12h downtrend filter
              volume_ratio[i] > 1.8):                    # Volume confirmation
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_12h_Donchian_Breakout_EMA21_Volume_Filter"
timeframe = "4h"
leverage = 1.0