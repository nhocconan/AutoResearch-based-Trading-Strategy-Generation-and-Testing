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
    
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate weekly ATR(14) for volatility filter
    tr1 = pd.Series(weekly_high - weekly_low)
    tr2 = pd.Series(np.abs(weekly_high - np.concatenate([[weekly_close[0]], weekly_close[:-1]])))
    tr3 = pd.Series(np.abs(weekly_low - np.concatenate([[weekly_close[0]], weekly_close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 12h timeframe with proper delay
    weekly_high_12h = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_12h = align_htf_to_ltf(prices, df_1w, weekly_low)
    atr_14_12h = align_htf_to_ltf(prices, df_1w, atr_14)
    
    # Calculate 12h Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_high_12h[i]) or np.isnan(weekly_low_12h[i]) or 
            np.isnan(atr_14_12h[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 12h price breaks above weekly high with volume confirmation → long
        # 2. 12h price breaks below weekly low with volume confirmation → short
        # 3. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        # 4. Volume confirmation: volume > 1.5x average
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: 12h breakout above weekly high
        if (close[i] > weekly_high_12h[i] and            # 12h price above weekly high
            volume_ratio[i] > 1.5 and                   # Volume confirmation
            atr_14_12h[i] > 0.005 * close[i]):          # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: 12h breakdown below weekly low
        elif (close[i] < weekly_low_12h[i] and          # 12h price below weekly low
              volume_ratio[i] > 1.5 and                 # Volume confirmation
              atr_14_12h[i] > 0.005 * close[i]):        # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Weekly_High_Low_Breakout_Volume_ATR_Filter"
timeframe = "12h"
leverage = 1.0