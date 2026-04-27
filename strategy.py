#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter and Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Volume filter: require volume > 1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC (active trading hours)
    hour = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hour >= 8) & (hour <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 30  # need 30 for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > 12h Donchian high + uptrend + volume + session
            if (close[i] > donchian_high_aligned[i] and 
                ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and  # rising trend
                volume_filter[i] and 
                session_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < 12h Donchian low + downtrend + volume + session
            elif (close[i] < donchian_low_aligned[i] and 
                  ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and  # falling trend
                  volume_filter[i] and 
                  session_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price < 12h Donchian low OR trend reversal
            if (close[i] < donchian_low_aligned[i] or 
                ema50_12h_aligned[i] < ema50_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price > 12h Donchian high OR trend reversal
            if (close[i] > donchian_high_aligned[i] or 
                ema50_12h_aligned[i] > ema50_12h_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian50_12hTrend_Volume_Session"
timeframe = "4h"
leverage = 1.0