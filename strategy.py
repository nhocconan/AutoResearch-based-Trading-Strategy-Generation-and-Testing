#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Keltner_Channel_Breakout_1wTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly trend (ATR-based EMA)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate ATR for Keltner Channel (using 1d data)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Channel parameters
    kc_mult = 2.0
    kc_ema = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    kc_upper = kc_ema + kc_mult * atr
    kc_lower = kc_ema - kc_mult * atr
    
    # Weekly trend filter: EMA20 on weekly close
    ema_20w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20w_aligned = align_htf_to_ltf(prices, df_1w, ema_20w)
    
    # Volume filter: 2x 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for EMA and ATR to stabilize
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(ema_20w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]  # Volume above average
        
        # Session filter: 00-24 UTC (full day for 1d timeframe)
        # For 1d, we consider the entire day active
        
        if position == 0:
            # Long: price breaks above upper Keltner + above weekly EMA (uptrend) + volume
            if (close[i] > kc_upper[i] and 
                close[i] > ema_20w_aligned[i] and 
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Keltner + below weekly EMA (downtrend) + volume
            elif (close[i] < kc_lower[i] and 
                  close[i] < ema_20w_aligned[i] and 
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below lower Keltner or volume drops significantly
            if close[i] < kc_lower[i] or volume[i] < 0.5 * vol_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above upper Keltner or volume drops significantly
            if close[i] > kc_upper[i] or volume[i] < 0.5 * vol_ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals