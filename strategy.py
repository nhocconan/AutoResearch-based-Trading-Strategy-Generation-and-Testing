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
    
    # Get daily data for ATR and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily ATR(14) for volatility
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, tr2)])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily volume average for volatility filter
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align daily indicators to 4h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate 4h price change for momentum
    price_change = np.diff(close, prepend=close[0])
    price_change_ma = pd.Series(price_change).rolling(window=10, min_periods=10).mean().values
    
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
            np.isnan(vol_avg_aligned[i]) or
            np.isnan(price_change_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR above average
        vol_filter = atr_aligned[i] > vol_avg_aligned[i]
        
        # Momentum filter: positive price change for long, negative for short
        mom_long = price_change_ma[i] > 0
        mom_short = price_change_ma[i] < 0
        
        # Entry conditions
        long_entry = vol_filter and mom_long
        short_entry = vol_filter and mom_short
        
        # Exit conditions: volatility contraction or momentum reversal
        long_exit = not vol_filter or not mom_long
        short_exit = not vol_filter or not mom_short
        
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

name = "4h_Volatility_Momentum_Filter"
timeframe = "4h"
leverage = 1.0