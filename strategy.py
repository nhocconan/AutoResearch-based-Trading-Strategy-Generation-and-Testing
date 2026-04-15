#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_volume = df_1d['volume'].values
    
    # Calculate 20-period daily Donchian channels
    highest_20 = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 14-period daily ATR for volatility filter
    daily_close_prev = np.concatenate([[daily_close[0]], daily_close[:-1]])
    tr = np.maximum(daily_high - daily_low,
                    np.maximum(np.abs(daily_high - daily_close_prev),
                               np.abs(daily_low - daily_close_prev)))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_30 = pd.Series(atr_14).rolling(window=30, min_periods=30).mean().values
    volatility_ratio = atr_14 / (atr_ma_30 + 1e-10)
    
    # Calculate daily RSI(14) for momentum filter
    delta = np.diff(daily_close, prepend=daily_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Align HTF indicators to 4h timeframe with proper delay
    highest_20_4h = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_4h = align_htf_to_ltf(prices, df_1d, lowest_20)
    volatility_ratio_4h = align_htf_to_ltf(prices, df_1d, volatility_ratio)
    rsi_14_4h = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20_4h[i]) or np.isnan(lowest_20_4h[i]) or 
            np.isnan(volatility_ratio_4h[i]) or np.isnan(rsi_14_4h[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Price breaks above/below daily Donchian(20) channel
        # 2. Volume confirmation (1.5x average)
        # 3. Avoid low volatility squeezes (volatility ratio > 0.7)
        # 4. RSI not extreme (30 < RSI < 70) to avoid chasing
        # 5. Discrete position sizing: 0.25
        
        # Long conditions
        if (close[i] > highest_20_4h[i] and     # Donchian breakout
            volume_ratio[i] > 1.5 and           # Volume confirmation
            volatility_ratio_4h[i] > 0.7 and    # Avoid low volatility
            30 < rsi_14_4h[i] < 70):            # RSI not extreme
            signals[i] = 0.25
            
        # Short conditions
        elif (close[i] < lowest_20_4h[i] and    # Donchian breakdown
              volume_ratio[i] > 1.5 and         # Volume confirmation
              volatility_ratio_4h[i] > 0.7 and  # Avoid low volatility
              30 < rsi_14_4h[i] < 70):          # RSI not extreme
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_DailyDonchian_Breakout_Volume_RSI_Filter"
timeframe = "4h"
leverage = 1.0