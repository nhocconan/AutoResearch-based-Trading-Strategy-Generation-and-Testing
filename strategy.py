#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Donchian_Breakout_Volume_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12-period EMA on 12h data for trend filter
    close_12h_series = pd.Series(close_12h)
    ema_12h = close_12h_series.ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # Align 12h EMA to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Donchian channels (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max().values
    lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 12)
    
    for i in range(start_idx, n):
        if np.isnan(ema_12h_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or \
           np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema_trend = ema_12h_aligned[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: price breaks above upper Donchian + uptrend + volume
            if price > upper[i] and price > ema_trend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + downtrend + volume
            elif price < lower[i] and price < ema_trend and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below lower Donchian or trend turns down
            if price < lower[i] or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above upper Donchian or trend turns up
            if price > upper[i] or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals