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
    
    # Calculate 20-period weekly Donchian channels
    weekly_highest_20 = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    weekly_lowest_20 = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 50-period weekly EMA for trend filter
    weekly_ema_50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to daily timeframe with proper delay
    weekly_ema_50_1d = align_htf_to_ltf(prices, df_1w, weekly_ema_50)
    weekly_highest_20_1d = align_htf_to_ltf(prices, df_1w, weekly_highest_20)
    weekly_lowest_20_1d = align_htf_to_ltf(prices, df_1w, weekly_lowest_20)
    
    # Calculate 20-period daily Donchian channels for entry
    daily_highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    daily_lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 14-period daily ATR for volatility filter
    daily_close_prev = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - daily_close_prev),
                               np.abs(low - daily_close_prev)))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    volatility_ratio = atr_14 / (atr_ma_50 + 1e-10)
    
    # Calculate 20-period daily volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_ema_50_1d[i]) or np.isnan(weekly_highest_20_1d[i]) or 
            np.isnan(weekly_lowest_20_1d[i]) or np.isnan(daily_highest_20[i]) or 
            np.isnan(daily_lowest_20[i]) or np.isnan(volume_ratio[i]) or 
            np.isnan(volatility_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Weekly trend filter: price above/below weekly EMA50
        # 2. Weekly structure: price near weekly Donchian channels
        # 3. Daily Donchian breakout with volume confirmation
        # 4. Volatility filter: avoid extremely low volatility
        # 5. Discrete position sizing: 0.25
        
        # Long conditions
        if (close[i] > weekly_ema_50_1d[i] and  # Weekly uptrend filter
            close[i] > weekly_highest_20_1d[i] * 0.98 and  # Near weekly high (within 2%)
            close[i] > daily_highest_20[i] and     # Daily Donchian breakout
            volume_ratio[i] > 1.5 and              # Volume confirmation
            volatility_ratio[i] > 0.5):            # Avoid extremely low volatility
            signals[i] = 0.25
            
        # Short conditions
        elif (close[i] < weekly_ema_50_1d[i] and   # Weekly downtrend filter
              close[i] < weekly_lowest_20_1d[i] * 1.02 and  # Near weekly low (within 2%)
              close[i] < daily_lowest_20[i] and      # Daily Donchian breakdown
              volume_ratio[i] > 1.5 and              # Volume confirmation
              volatility_ratio[i] > 0.5):            # Avoid extremely low volatility
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyEMA_Donchian_Breakout_Volume_Filter"
timeframe = "1d"
leverage = 1.0