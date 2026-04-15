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
    weekly_volume = df_1w['volume'].values
    
    # Calculate 20-period weekly Donchian channels
    weekly_highest_20 = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    weekly_lowest_20 = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 14-period weekly ATR for volatility filter
    weekly_close_prev = np.concatenate([[weekly_close[0]], weekly_close[:-1]])
    weekly_tr = np.maximum(weekly_high - weekly_low,
                           np.maximum(np.abs(weekly_high - weekly_close_prev),
                                      np.abs(weekly_low - weekly_close_prev)))
    weekly_atr_14 = pd.Series(weekly_tr).rolling(window=14, min_periods=14).mean().values
    weekly_atr_ma_50 = pd.Series(weekly_atr_14).rolling(window=50, min_periods=50).mean().values
    weekly_volatility_ratio = weekly_atr_14 / (weekly_atr_ma_50 + 1e-10)
    
    # Calculate 50-period weekly EMA for trend filter
    weekly_ema_50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to daily timeframe with proper delay
    weekly_ema_50_1d = align_htf_to_ltf(prices, df_1w, weekly_ema_50)
    weekly_highest_20_1d = align_htf_to_ltf(prices, df_1w, weekly_highest_20)
    weekly_lowest_20_1d = align_htf_to_ltf(prices, df_1w, weekly_lowest_20)
    weekly_volatility_ratio_1d = align_htf_to_ltf(prices, df_1w, weekly_volatility_ratio)
    
    # Calculate daily volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_ema_50_1d[i]) or np.isnan(weekly_highest_20_1d[i]) or 
            np.isnan(weekly_lowest_20_1d[i]) or np.isnan(weekly_volatility_ratio_1d[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Weekly trend filter: price above/below weekly EMA50
        # 2. Weekly volatility regime: only trade in normal/high volatility (avoid low vol squeezes)
        # 3. Weekly Donchian breakout/breakdown with volume confirmation
        # 4. Discrete position sizing: 0.25
        
        # Long conditions
        if (close[i] > weekly_ema_50_1d[i] and  # Uptrend filter
            weekly_volatility_ratio_1d[i] > 0.8 and  # Avoid low volatility squeezes
            close[i] > weekly_highest_20_1d[i] and     # Weekly Donchian breakout
            volume_ratio[i] > 1.5):                    # Volume confirmation
            signals[i] = 0.25
            
        # Short conditions
        elif (close[i] < weekly_ema_50_1d[i] and   # Downtrend filter
              weekly_volatility_ratio_1d[i] > 0.8 and  # Avoid low volatility squeezes
              close[i] < weekly_lowest_20_1d[i] and      # Weekly Donchian breakdown
              volume_ratio[i] > 1.5):                    # Volume confirmation
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyEMA_Volatility_Donchian_Breakout"
timeframe = "1d"
leverage = 1.0