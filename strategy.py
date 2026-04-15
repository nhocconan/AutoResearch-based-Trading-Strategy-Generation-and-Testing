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
    
    # Calculate weekly Donchian channels (20-period)
    highest_20w = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    lowest_20w = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly ATR for volatility filter
    weekly_close_prev = np.concatenate([[weekly_close[0]], weekly_close[:-1]])
    tr_w = np.maximum(weekly_high - weekly_low,
                      np.maximum(np.abs(weekly_high - weekly_close_prev),
                                 np.abs(weekly_low - weekly_close_prev)))
    atr_14w = pd.Series(tr_w).rolling(window=14, min_periods=14).mean().values
    atr_ma_50w = pd.Series(atr_14w).rolling(window=50, min_periods=50).mean().values
    volatility_ratio_w = atr_14w / (atr_ma_50w + 1e-10)
    
    # Calculate weekly EMA(50) for trend filter
    ema_50w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 12h timeframe with proper delay
    highest_20w_12h = align_htf_to_ltf(prices, df_1w, highest_20w)
    lowest_20w_12h = align_htf_to_ltf(prices, df_1w, lowest_20w)
    ema_50w_12h = align_htf_to_ltf(prices, df_1w, ema_50w)
    volatility_ratio_w_12h = align_htf_to_ltf(prices, df_1w, volatility_ratio_w)
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20w_12h[i]) or np.isnan(lowest_20w_12h[i]) or 
            np.isnan(ema_50w_12h[i]) or np.isnan(volatility_ratio_w_12h[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Weekly trend filter: price above/below weekly EMA50
        # 2. Weekly volatility regime: only trade in normal/high volatility (avoid low vol squeezes)
        # 3. Weekly Donchian breakout/breakdown with volume confirmation
        # 4. Discrete position sizing: 0.25
        
        # Long conditions
        if (close[i] > ema_50w_12h[i] and        # Uptrend filter
            volatility_ratio_w_12h[i] > 0.8 and   # Avoid low volatility squeezes
            close[i] > highest_20w_12h[i] and     # Weekly Donchian breakout
            volume_ratio[i] > 1.5):               # Volume confirmation
            signals[i] = 0.25
            
        # Short conditions
        elif (close[i] < ema_50w_12h[i] and      # Downtrend filter
              volatility_ratio_w_12h[i] > 0.8 and # Avoid low volatility squeezes
              close[i] < lowest_20w_12h[i] and    # Weekly Donchian breakdown
              volume_ratio[i] > 1.5):             # Volume confirmation
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_WeeklyEMA_Volatility_Donchian_Breakout"
timeframe = "12h"
leverage = 1.0