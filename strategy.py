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
    
    # Get weekly data for trend context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ATR(14)
    tr1 = np.maximum(high_1w[1:], low_1w[:-1]) - np.minimum(high_1w[1:], low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate weekly SMA(50)
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    
    # Align weekly indicators to daily
    atr_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    sma50_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
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
        atr_ma_1w = pd.Series(atr_1w).rolling(window=10, min_periods=10).mean().values
        atr_ma_aligned = align_htf_to_ltf(prices, df_1w, atr_ma_1w)
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

name = "1d_SMA50_ATR14_Volume_Trend_Session"
timeframe = "1d"
leverage = 1.0