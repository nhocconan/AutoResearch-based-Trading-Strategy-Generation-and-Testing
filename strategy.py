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
    
    # Get daily data for ATR and moving averages
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = np.maximum(high_1d[1:], low_1d[:-1]) - np.minimum(high_1d[1:], low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 6-day EMA for trend filter (shorter for responsiveness)
    ema6_1d = pd.Series(close_1d).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    # Calculate 14-day EMA for trend filter (longer for confirmation)
    ema14_1d = pd.Series(close_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate daily volume average (20-period)
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 12h
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema6_aligned = align_htf_to_ltf(prices, df_1d, ema6_1d)
    ema14_aligned = align_htf_to_ltf(prices, df_1d, ema14_1d)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_aligned[i]) or 
            np.isnan(ema6_aligned[i]) or 
            np.isnan(ema14_aligned[i]) or
            np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: only trade when volume is above average
        vol_current = volume[i]
        vol_filter = vol_current > vol_avg_aligned[i]
        
        # Trend filter: EMA6 > EMA14 for uptrend, EMA6 < EMA14 for downtrend
        uptrend = ema6_aligned[i] > ema14_aligned[i]
        downtrend = ema6_aligned[i] < ema14_aligned[i]
        
        # Long conditions: uptrend + volume filter + price above EMA6
        long_condition = uptrend and vol_filter and close[i] > ema6_aligned[i]
        
        # Short conditions: downtrend + volume filter + price below EMA6
        short_condition = downtrend and vol_filter and close[i] < ema6_aligned[i]
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal or volume filter failure
        elif position == 1 and (not uptrend or not vol_filter):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not downtrend or not vol_filter):
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

name = "12h_EMA6_EMA14_VolumeFilter_Trend"
timeframe = "12h"
leverage = 1.0