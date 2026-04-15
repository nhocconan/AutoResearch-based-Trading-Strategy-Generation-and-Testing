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
    
    # Calculate daily Williams %R (14-period)
    highest_high_14 = pd.Series(daily_high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(daily_low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - daily_close) / (highest_high_14 - lowest_low_14 + 1e-10)
    
    # Align HTF indicators to 6h timeframe with proper delay
    williams_r_6h = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 6h Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_6h[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 6h price breaks above Donchian(20) high with volume confirmation → long
        # 2. 6h price breaks below Donchian(20) low with volume confirmation → short
        # 3. Williams %R filter: avoid extreme overbought/oversold for continuation
        #    Long: Williams %R > -80 (not deeply oversold)
        #    Short: Williams %R < -20 (not deeply overbought)
        # 4. Volume confirmation: volume > 1.5x average
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: 6h breakout above Donchian high
        if (close[i] > highest_20[i] and            # 6h price above Donchian high
            williams_r_6h[i] > -80.0 and           # Not deeply oversold
            volume_ratio[i] > 1.5):                # Volume confirmation
            signals[i] = 0.25
            
        # Short conditions: 6h breakdown below Donchian low
        elif (close[i] < lowest_20[i] and          # 6h price below Donchian low
              williams_r_6h[i] < -20.0 and         # Not deeply overbought
              volume_ratio[i] > 1.5):              # Volume confirmation
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR_Donchian_Breakout_Volume_Filter"
timeframe = "6h"
leverage = 1.0