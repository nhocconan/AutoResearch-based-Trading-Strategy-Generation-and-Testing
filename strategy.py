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
    
    # Get 12h HTF data once before loop (as per experiment instructions)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(df_12h['high'].values).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(df_12h['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h ATR (14-period) for volatility filter and stoploss
    high_low = df_12h['high'].values - df_12h['low'].values
    high_close_prev = np.abs(df_12h['high'].values - np.concatenate([[df_12h['close'].values[0]], df_12h['close'].values[:-1]]))
    low_close_prev = np.abs(df_12h['low'].values - np.concatenate([[df_12h['close'].values[0]], df_12h['close'].values[:-1]]))
    tr_12h = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
    atr_14 = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    volatility_ratio = atr_14 / (atr_ma_50 + 1e-10)
    
    # Calculate 12h EMA (50-period) for trend filter
    ema_50 = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 4h timeframe with proper delay
    highest_20_4h = align_htf_to_ltf(prices, df_12h, highest_20)
    lowest_20_4h = align_htf_to_ltf(prices, df_12h, lowest_20)
    ema_50_4h = align_htf_to_ltf(prices, df_12h, ema_50)
    volatility_ratio_4h = align_htf_to_ltf(prices, df_12h, volatility_ratio)
    
    # Calculate 4h volume ratio (current vs 20-period average) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20_4h[i]) or np.isnan(lowest_20_4h[i]) or 
            np.isnan(ema_50_4h[i]) or np.isnan(volatility_ratio_4h[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 12h Donchian breakout/breakdown
        # 2. 12h EMA50 trend filter
        # 3. Volatility regime: avoid extremely low volatility (choppy markets)
        # 4. Volume confirmation
        # 5. Discrete position sizing: 0.25
        
        # Long conditions: price breaks above 12h Donchian high + uptrend + volume + volatility filter
        if (close[i] > highest_20_4h[i] and     # 12h Donchian breakout
            close[i] > ema_50_4h[i] and         # Above 12h EMA50 (uptrend)
            volatility_ratio_4h[i] > 0.6 and    # Avoid extremely low volatility
            volume_ratio[i] > 1.3):             # Volume confirmation
            signals[i] = 0.25
            
        # Short conditions: price breaks below 12h Donchian low + downtrend + volume + volatility filter
        elif (close[i] < lowest_20_4h[i] and    # 12h Donchian breakdown
              close[i] < ema_50_4h[i] and       # Below 12h EMA50 (downtrend)
              volatility_ratio_4h[i] > 0.6 and  # Avoid extremely low volatility
              volume_ratio[i] > 1.3):           # Volume confirmation
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_12h_Donchian_Breakout_EMA50_Volume_VolatilityFilter"
timeframe = "4h"
leverage = 1.0