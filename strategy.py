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
    
    # Calculate daily EMA(50) for trend filter
    ema_50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = pd.Series(daily_high - daily_low)
    tr2 = pd.Series(np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr3 = pd.Series(np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 6h timeframe with proper delay
    williams_r_6h = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50)
    atr_14_6h = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 6h Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_6h[i]) or np.isnan(ema_50_6h[i]) or np.isnan(atr_14_6h[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Daily Williams %R extreme (oversold/overbought)
        # 2. 6h Donchian breakout in direction of Williams %R signal
        # 3. Daily trend filter: price above/below daily EMA50
        # 4. 6h volume confirmation: volume > 1.5x average
        # 5. 6h volatility filter: ATR > 0.3% of price (avoid low volatility chop)
        # 6. Discrete position sizing: 0.25
        
        # Long conditions: Williams %R oversold (< -80) + Donchian breakout above in uptrend
        if (williams_r_6h[i] < -80 and          # Daily oversold
            close[i] > ema_50_6h[i] and         # Daily uptrend filter
            close[i] > highest_20[i] and        # 6h Donchian breakout
            volume_ratio[i] > 1.5 and           # Volume confirmation
            atr_14_6h[i] > 0.003 * close[i]):   # Volatility filter
            signals[i] = 0.25
            
        # Short conditions: Williams %R overbought (> -20) + Donchian breakdown below in downtrend
        elif (williams_r_6h[i] > -20 and        # Daily overbought
              close[i] < ema_50_6h[i] and       # Daily downtrend filter
              close[i] < lowest_20[i] and       # 6h Donchian breakdown
              volume_ratio[i] > 1.5 and         # Volume confirmation
              atr_14_6h[i] > 0.003 * close[i]): # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR_Donchian_Breakout_EMA50_Volume_ATR_Filter"
timeframe = "6h"
leverage = 1.0