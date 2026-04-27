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
    
    # Get 4h data for trend context
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA 50 for trend direction
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA 200 for long-term trend
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 1h Donchian channels (20-period for breakout)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        # Skip outside session
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        # Trend filters
        long_term_bull = close[i] > ema_200_1d_aligned[i]
        medium_term_bull = close[i] > ema_50_4h_aligned[i]
        long_term_bear = close[i] < ema_200_1d_aligned[i]
        medium_term_bear = close[i] < ema_50_4h_aligned[i]
        
        # Long conditions: breakout above upper Donchian + medium-term bullish + long-term bullish + volume
        long_breakout = (close[i] > highest_high[i-1] and 
                        medium_term_bull and long_term_bull and volume_filter[i])
        
        # Short conditions: breakout below lower Donchian + medium-term bearish + long-term bearish + volume
        short_breakout = (close[i] < lowest_low[i-1] and 
                         medium_term_bear and long_term_bear and volume_filter[i])
        
        if long_breakout:
            signals[i] = 0.20
            position = 1
        elif short_breakout:
            signals[i] = -0.20
            position = -1
        # Exit conditions: opposite Donchian breakout or trend reversal
        elif position == 1 and (close[i] < lowest_low[i-1] or close[i] < ema_50_4h_aligned[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > highest_high[i-1] or close[i] > ema_50_4h_aligned[i]):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_Donchian20_4hEMA50_1dEMA200_VolumeSessionFilter"
timeframe = "1h"
leverage = 1.0