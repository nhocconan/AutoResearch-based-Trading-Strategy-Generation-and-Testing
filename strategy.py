#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Time-Based Momentum with 4h Trend Filter and Volume Confirmation
# Uses 4h EMA trend direction and daily volume profile for bias, enters on 1h momentum bursts
# during active sessions (08-20 UTC) with volume confirmation. Works in bull/bear by
# following 4h trend. Target: 60-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute hour filter for session (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data for trend filter (EMA21)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Load daily data for volume average (20-day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    vol_20d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_20d_aligned = align_htf_to_ltf(prices, df_1d, vol_20d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.20  # Position size
    
    for i in range(100, n):
        # Skip if outside trading session
        if not in_session[i]:
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(vol_20d_aligned[i])):
            continue
        
        # Long: price above 4h EMA (uptrend) + volume spike + upward momentum
        if (close[i] > ema_4h_aligned[i] and
            volume[i] > 1.5 * vol_20d_aligned[i] and
            close[i] > close[i-1] and
            close[i-1] > close[i-2] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short: price below 4h EMA (downtrend) + volume spike + downward momentum
        elif (close[i] < ema_4h_aligned[i] and
              volume[i] > 1.5 * vol_20d_aligned[i] and
              close[i] < close[i-1] and
              close[i-1] < close[i-2] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: trend reversal or momentum fade
        elif position == 1 and (close[i] < ema_4h_aligned[i] or 
                                (close[i] < close[i-1] and close[i-1] < close[i-2])):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > ema_4h_aligned[i] or 
                                 (close[i] > close[i-1] and close[i-1] > close[i-2])):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1h_Trend_Momentum_Volume_Session"
timeframe = "1h"
leverage = 1.0