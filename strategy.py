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
    
    # Get daily HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_volume = df_1d['volume'].values
    
    # Calculate daily Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = pd.Series(daily_high - daily_low)
    tr2 = pd.Series(np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr3 = pd.Series(np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 1d timeframe with proper delay
    highest_20_1d = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_1d = align_htf_to_ltf(prices, df_1d, lowest_20)
    atr_14_1d = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 1d volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20_1d[i]) or np.isnan(lowest_20_1d[i]) or 
            np.isnan(atr_14_1d[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 1d price breaks above 20-day high with volume confirmation → long
        # 2. 1d price breaks below 20-day low with volume confirmation → short
        # 3. Volatility filter: ATR > 0.3% of price (avoid low volatility chop)
        # 4. Volume confirmation: volume > 1.5x average
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: 1d breakout above 20-day high
        if (close[i] > highest_20_1d[i] and            # 1d price above 20-day high
            volume_ratio[i] > 1.5 and                  # Volume confirmation
            atr_14_1d[i] > 0.003 * close[i]):          # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: 1d breakdown below 20-day low
        elif (close[i] < lowest_20_1d[i] and           # 1d price below 20-day low
              volume_ratio[i] > 1.5 and                # Volume confirmation
              atr_14_1d[i] > 0.003 * close[i]):        # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_Breakout_Volume_ATR_Filter"
timeframe = "1d"
leverage = 1.0