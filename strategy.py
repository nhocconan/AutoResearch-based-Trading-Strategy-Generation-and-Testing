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
    
    # Calculate 20-period daily Donchian channels
    highest_20d = pd.Series(daily_high).rolling(window=20, min_periods=20).max().values
    lowest_20d = pd.Series(daily_low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 50-period daily EMA for trend filter
    ema_50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily ATR(14) for volatility filter
    daily_close_prev = np.concatenate([[daily_close[0]], daily_close[:-1]])
    tr = np.maximum(daily_high - daily_low,
                    np.maximum(np.abs(daily_high - daily_close_prev),
                               np.abs(daily_low - daily_close_prev)))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr_14 > (0.5 * atr_ma_50)  # Trade only when ATR > 50% of its MA
    
    # Align HTF indicators to 4h timeframe
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50)
    highest_20d_4h = align_htf_to_ltf(prices, df_1d, highest_20d)
    lowest_20d_4h = align_htf_to_ltf(prices, df_1d, lowest_20d)
    volatility_filter_4h = align_htf_to_ltf(prices, df_1d, volatility_filter)
    
    # Calculate 4h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h[i]) or np.isnan(highest_20d_4h[i]) or 
            np.isnan(lowest_20d_4h[i]) or np.isnan(volatility_filter_4h[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Long: 4h price breaks above daily Donchian high + volume + volatility + trend
        if (close[i] > highest_20d_4h[i] and     # Break above daily Donchian high
            volume_ratio[i] > 1.5 and           # Volume confirmation
            volatility_filter_4h[i] and         # Sufficient volatility
            ema_50_4h[i] > 0):                  # Valid EMA (trend filter)
            signals[i] = 0.25
            
        # Short: 4h price breaks below daily Donchian low + volume + volatility + trend
        elif (close[i] < lowest_20d_4h[i] and    # Break below daily Donchian low
              volume_ratio[i] > 1.5 and          # Volume confirmation
              volatility_filter_4h[i] and        # Sufficient volatility
              ema_50_4h[i] > 0):                 # Valid EMA (trend filter)
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_DailyDonchian_Breakout_Volume_Volatility_Trend"
timeframe = "4h"
leverage = 1.0