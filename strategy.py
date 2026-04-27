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
    
    # Get weekly data for higher timeframe context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 50-week EMA for long-term trend
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get daily data for intermediate context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 34-day EMA for intermediate trend
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 14-day ATR for volatility filter
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]), np.abs(low_1d[1:] - close_1d[:-1]))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 6-hour Donchian channels (15-period for structure)
    # Use 6h data from prices directly (primary timeframe)
    highest_high_6h = pd.Series(high).rolling(window=15, min_periods=15).max().values
    lowest_low_6h = pd.Series(low).rolling(window=15, min_periods=15).min().values
    
    # Volume filter: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(highest_high_6h[i]) or 
            np.isnan(lowest_low_6h[i]) or 
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        # Multi-timeframe trend alignment: both weekly and daily trends must agree
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]
        daily_uptrend = close[i] > ema_34_1d_aligned[i]
        daily_downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volatility filter: only trade when volatility is elevated (avoid choppy markets)
        vol_filter = atr_1d_aligned[i] > np.nanmedian(atr_1d_aligned[max(0, i-30):i+1])
        
        # Long conditions: price breaks above 6h Donchian high + multi-timeframe uptrend + volume + volatility
        long_breakout = (close[i] > highest_high_6h[i-1] and 
                        weekly_uptrend and 
                        daily_uptrend and 
                        volume_filter[i] and 
                        vol_filter)
        
        # Short conditions: price breaks below 6h Donchian low + multi-timeframe downtrend + volume + volatility
        short_breakout = (close[i] < lowest_low_6h[i-1] and 
                         weekly_downtrend and 
                         daily_downtrend and 
                         volume_filter[i] and 
                         vol_filter)
        
        if long_breakout:
            signals[i] = 0.25
            position = 1
        elif short_breakout:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite 6h Donchian breakout
        elif position == 1 and close[i] < lowest_low_6h[i-1]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > highest_high_6h[i-1]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Donchian15_1w50EMA_1d34EMA_VolumeFilter"
timeframe = "6h"
leverage = 1.0