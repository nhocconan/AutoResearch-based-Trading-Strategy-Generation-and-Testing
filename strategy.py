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
    
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Calculate 20-period weekly ATR for volatility regime filter
    weekly_close_prev = np.concatenate([[weekly_close[0]], weekly_close[:-1]])
    tr = np.maximum(weekly_high - weekly_low,
                    np.maximum(np.abs(weekly_high - weekly_close_prev),
                               np.abs(weekly_low - weekly_close_prev)))
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    atr_ma_50 = pd.Series(atr_20).rolling(window=50, min_periods=50).mean().values
    volatility_ratio = atr_20 / (atr_ma_50 + 1e-10)
    
    # Calculate 50-period weekly EMA for trend filter
    ema_50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1d timeframe with proper delay
    ema_50_1d = align_htf_to_ltf(prices, df_1w, ema_50)
    volatility_ratio_1d = align_htf_to_ltf(prices, df_1w, volatility_ratio)
    
    # Calculate 1d Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d[i]) or np.isnan(volatility_ratio_1d[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Weekly trend filter: price above/below weekly EMA50
        # 2. Volatility regime: only trade in normal/high volatility (avoid low vol squeezes)
        # 3. 1d Donchian breakout with volume confirmation
        # 4. Discrete position sizing: 0.25
        
        # Long conditions
        if (close[i] > ema_50_1d[i] and          # Uptrend filter
            volatility_ratio_1d[i] > 0.8 and     # Avoid low volatility squeezes
            close[i] > highest_20[i] and         # Donchian breakout
            volume_ratio[i] > 1.5):              # Volume confirmation
            signals[i] = 0.25
            
        # Short conditions
        elif (close[i] < ema_50_1d[i] and        # Downtrend filter
              volatility_ratio_1d[i] > 0.8 and   # Avoid low volatility squeezes
              close[i] < lowest_20[i] and         # Donchian breakdown
              volume_ratio[i] > 1.5):             # Volume confirmation
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyEMA_Volume_Donchian_Breakout"
timeframe = "1d"
leverage = 1.0