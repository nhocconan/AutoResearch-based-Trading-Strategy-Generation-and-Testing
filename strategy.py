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
    
    # Calculate daily Donchian channels (20-period)
    highest_high_20 = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = pd.Series(daily_high - daily_low)
    tr2 = pd.Series(np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr3 = pd.Series(np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 12h timeframe with proper delay
    highest_high_20_12h = align_htf_to_ltf(prices, df_1d, highest_high_20)
    lowest_low_20_12h = align_htf_to_ltf(prices, df_1d, lowest_low_20)
    atr_14_12h = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 12h Donchian channels (10-period) for breakout signals
    highest_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    lowest_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high_20_12h[i]) or np.isnan(lowest_low_20_12h[i]) or np.isnan(atr_14_12h[i]) or 
            np.isnan(highest_10[i]) or np.isnan(lowest_10[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Daily Donchian breakout (20-period)
        # 2. 12h Donchian breakout in same direction (10-period) for confirmation
        # 3. Volume confirmation: volume > 2.0x average
        # 4. Volatility filter: ATR > 0.5% of price (avoid low volatility chop)
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: Daily Donchian breakout above + 12h confirmation
        if (close[i] > highest_high_20_12h[i] and      # Daily Donchian breakout (20)
            close[i] > highest_10[i] and                # 12h Donchian breakout (10) confirmation
            volume_ratio[i] > 2.0 and                   # Volume confirmation
            atr_14_12h[i] > 0.005 * close[i]):          # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: Daily Donchian breakdown below + 12h confirmation
        elif (close[i] < lowest_low_20_12h[i] and     # Daily Donchian breakdown (20)
              close[i] < lowest_10[i] and               # 12h Donchian breakdown (10) confirmation
              volume_ratio[i] > 2.0 and                 # Volume confirmation
              atr_14_12h[i] > 0.005 * close[i]):        # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_DailyDonchian20_12hDonchian10_Volume_ATR_Filter"
timeframe = "12h"
leverage = 1.0