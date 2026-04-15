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
    
    # Get weekly HTF data once before loop (6h primary, 1w HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Calculate 6h Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Calculate 1w Williams %R (14-period) for overbought/oversold
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(weekly_high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(weekly_low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - weekly_close) / (highest_high_14 - lowest_low_14 + 1e-10) * -100
    
    # Calculate 1w EMA(50) for trend filter
    ema_50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe with proper delay
    ema_50_6h = align_htf_to_ltf(prices, df_1w, ema_50)
    williams_r_6h = align_htf_to_ltf(prices, df_1w, williams_r)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_6h[i]) or np.isnan(williams_r_6h[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 1w trend filter: price above/below weekly EMA50
        # 2. 1w Williams %R filter: avoid extreme readings
        # 3. 6h Donchian breakout: price breaks 20-period channel
        # 4. 6h volume confirmation: volume > 1.3x average
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: break above Donchian high in uptrend, not overbought
        if (close[i] > ema_50_6h[i] and          # Weekly uptrend filter
            williams_r_6h[i] > -80 and           # Not oversold (Williams %R > -80)
            close[i] > highest_20[i] and         # Donchian breakout high
            volume_ratio[i] > 1.3):              # Volume confirmation
            signals[i] = 0.25
            
        # Short conditions: break below Donchian low in downtrend, not oversold
        elif (close[i] < ema_50_6h[i] and        # Weekly downtrend filter
              williams_r_6h[i] < -20 and         # Not overbought (Williams %R < -20)
              close[i] < lowest_20[i] and        # Donchian breakdown low
              volume_ratio[i] > 1.3):            # Volume confirmation
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR_Donchian_Breakout_Volume_Trend"
timeframe = "6h"
leverage = 1.0