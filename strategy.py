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
    
    # Calculate 20-period daily Donchian channels for structure
    highest_20 = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 14-period daily ATR for volatility filter
    daily_close_prev = np.concatenate([[daily_close[0]], daily_close[:-1]])
    tr = np.maximum(daily_high - daily_low,
                    np.maximum(np.abs(daily_high - daily_close_prev),
                               np.abs(daily_low - daily_close_prev)))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    volatility_ratio = atr_14 / (atr_ma_50 + 1e-10)
    
    # Calculate 50-period daily EMA for trend filter
    ema_50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 4h timeframe with proper delay
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50)
    highest_20_4h = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_4h = align_htf_to_ltf(prices, df_1d, lowest_20)
    volatility_ratio_4h = align_htf_to_ltf(prices, df_1d, volatility_ratio)
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h[i]) or np.isnan(highest_20_4h[i]) or 
            np.isnan(lowest_20_4h[i]) or np.isnan(volatility_ratio_4h[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Daily trend filter: price above/below daily EMA50
        # 2. Volatility regime: only trade in normal/high volatility (avoid low vol squeezes)
        # 3. 4h price touching or breaking daily Donchian channels with volume confirmation
        # 4. Discrete position sizing: 0.25
        
        # Long conditions: price touches/breaks above daily upper Donchian in uptrend
        if (close[i] >= highest_20_4h[i] and     # Touch/break above daily upper Donchian
            close[i] > ema_50_4h[i] and          # Uptrend filter
            volatility_ratio_4h[i] > 0.8 and     # Avoid low volatility squeezes
            volume_ratio[i] > 1.5):              # Volume confirmation
            signals[i] = 0.25
            
        # Short conditions: price touches/breaks below daily lower Donchian in downtrend
        elif (close[i] <= lowest_20_4h[i] and    # Touch/break below daily lower Donchian
              close[i] < ema_50_4h[i] and        # Downtrend filter
              volatility_ratio_4h[i] > 0.8 and   # Avoid low volatility squeezes
              volume_ratio[i] > 1.5):            # Volume confirmation
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_DailyDonchian_Touch_Trend_Volume"
timeframe = "4h"
leverage = 1.0