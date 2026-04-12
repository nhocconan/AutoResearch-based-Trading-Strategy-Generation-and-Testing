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
    
    # Get 4h data for trend direction and volatility filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h EMA(20) for trend filter
    close_4h_series = pd.Series(close_4h)
    ema_20_4h = close_4h_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 4h ATR(14) for volatility filter
    tr_4h = np.maximum(high_4h[1:] - low_4h[1:], 
                       np.maximum(np.abs(high_4h[1:] - close_4h[:-1]), 
                                  np.abs(low_4h[1:] - close_4h[:-1])))
    tr_4h = np.concatenate([[np.nan], tr_4h])
    atr_14_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    atr_14_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_14_4h)
    
    # Get 1d data for Donchian channel breakout
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Donchian(20) channels
    highest_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    highest_20_1d_aligned = align_htf_to_ltf(prices, df_1d, highest_20_1d)
    lowest_20_1d_aligned = align_htf_to_ltf(prices, df_1d, lowest_20_1d)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if data not ready
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(atr_14_4h_aligned[i]) or
            np.isnan(highest_20_1d_aligned[i]) or np.isnan(lowest_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 4h EMA(20)
        uptrend = close[i] > ema_20_4h_aligned[i]
        downtrend = close[i] < ema_20_4h_aligned[i]
        
        # Volatility filter: avoid low volatility periods
        vol_filter = atr_14_4h_aligned[i] > 0
        
        # Donchian breakout conditions
        long_breakout = close[i] > highest_20_1d_aligned[i]
        short_breakout = close[i] < lowest_20_1d_aligned[i]
        
        # Entry conditions
        long_entry = long_breakout and uptrend and vol_filter
        short_entry = short_breakout and downtrend and vol_filter
        
        # Exit conditions: opposite Donchian break
        long_exit = close[i] < lowest_20_1d_aligned[i]
        short_exit = close[i] > highest_20_1d_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_donchian_breakout_ema_vol_filter_v1"
timeframe = "1h"
leverage = 1.0