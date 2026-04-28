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
    
    # Get daily data for ATR and SMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14)
    tr1 = np.maximum(high_1d[1:], low_1d[:-1]) - np.minimum(high_1d[1:], low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily SMA(50)
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align daily indicators to 4h
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    sma50_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    # Calculate average volume over 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_aligned[i]) or 
            np.isnan(sma50_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below SMA50
        uptrend = close[i] > sma50_aligned[i]
        downtrend = close[i] < sma50_aligned[i]
        
        # Volatility filter: only trade when ATR is above its 10-period average
        atr_ma_1d = pd.Series(atr_1d).rolling(window=10, min_periods=10).mean().values
        atr_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
        vol_filter = atr_aligned[i] > atr_ma_aligned[i] if not np.isnan(atr_ma_aligned[i]) else False
        
        # Volume filter: current volume above average
        vol_filter = vol_filter and volume[i] > vol_ma[i]
        
        # Entry conditions: trend + volatility + volume
        long_entry = uptrend and vol_filter
        short_entry = downtrend and vol_filter
        
        # Exit conditions: trend reversal or volatility drop
        long_exit = not uptrend or not vol_filter
        short_exit = not downtrend or not vol_filter
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_SMA50_ATR14_Volume_Trend_Session"
timeframe = "4h"
leverage = 1.0