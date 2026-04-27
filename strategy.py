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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA(20) for trend direction
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Get daily data for Donchian channel and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily Donchian(20) channel
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily volume moving average for volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to daily timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need all indicators
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_trend = ema_20_1w_aligned[i]
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter: current volume > 20-day average
        vol_filter = volume[i] > vol_ma if not np.isnan(vol_ma) else False
        
        if position == 0:
            # Long: price breaks above upper Donchian with up-trend and volume
            if close[i] > upper_val and close[i] > ema_trend and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below lower Donchian with down-trend and volume
            elif close[i] < lower_val and close[i] < ema_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below lower Donchian or trend turns down
            if close[i] < lower_val or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above upper Donchian or trend turns up
            if close[i] > upper_val or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_WeeklyEMA20_Trend_Volume"
timeframe = "1d"
leverage = 1.0