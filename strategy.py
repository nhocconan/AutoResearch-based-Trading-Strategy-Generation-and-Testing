#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Stochastic_Divergence_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and stochastic
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period stochastic on daily
    lookback = 14
    lowest_low = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().values
    highest_high = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().values
    stoch_k = np.where((highest_high - lowest_low) > 0, 
                       (close_1d - lowest_low) / (highest_high - lowest_low) * 100, 50)
    stoch_d = pd.Series(stoch_k).rolling(window=3, min_periods=3).mean().values
    
    # Align stochastic to 6h
    stoch_k_aligned = align_htf_to_ltf(prices, df_1d, stoch_k)
    stoch_d_aligned = align_htf_to_ltf(prices, df_1d, stoch_d)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation - 24-period average volume (6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(stoch_k_aligned[i]) or np.isnan(stoch_d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish divergence (stoch rising from oversold) + above EMA50 + volume
            if (stoch_k_aligned[i] > stoch_d_aligned[i] and 
                stoch_k_aligned[i-1] <= stoch_d_aligned[i-1] and  # crossover up
                stoch_k_aligned[i] < 30 and  # oversold
                close[i] > ema_50_1d_aligned[i] and
                vol_ratio[i] > 1.3):
                signals[i] = 0.25
                position = 1
            # Short: bearish divergence (stoch falling from overbought) + below EMA50 + volume
            elif (stoch_k_aligned[i] < stoch_d_aligned[i] and 
                  stoch_k_aligned[i-1] >= stoch_d_aligned[i-1] and  # crossover down
                  stoch_k_aligned[i] > 70 and  # overbought
                  close[i] < ema_50_1d_aligned[i] and
                  vol_ratio[i] > 1.3):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish crossover or below EMA50
            if (stoch_k_aligned[i] < stoch_d_aligned[i] and 
                stoch_k_aligned[i-1] >= stoch_d_aligned[i-1]) or \
               close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish crossover or above EMA50
            if (stoch_k_aligned[i] > stoch_d_aligned[i] and 
                stoch_k_aligned[i-1] <= stoch_d_aligned[i-1]) or \
               close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals