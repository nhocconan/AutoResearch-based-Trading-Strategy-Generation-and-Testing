#!/usr/bin/env python3
# 1d_ma_crossover_volume_filter
# Hypothesis: 1-day SMA crossover (50/200) with volume confirmation and volatility filter.
# Designed to work in both bull and bear markets by filtering for high-volume, trending periods.
# Target: 8-12 trades/year for minimal fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_ma_crossover_volume_filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week trend filter (SMA200) - load once before loop
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    sma200_1w = pd.Series(close_1w).rolling(window=200, min_periods=200).mean().values
    sma200_1w_aligned = align_htf_to_ltf(prices, df_1w, sma200_1w)
    
    # 1-day indicators
    sma50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    sma200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    atr = pd.Series(np.maximum(np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1])), np.abs(low[1:] - close[:-1]))).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr = np.concatenate([[np.nan], atr])  # Align length
    
    # Volume confirmation (volume > 1.5x 20-day average)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 200  # Need indicators warmed up
    
    for i in range(start_idx, n):
        if np.isnan(sma50[i]) or np.isnan(sma200[i]) or np.isnan(atr[i]) or np.isnan(avg_volume[i]) or np.isnan(sma200_1w_aligned[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_uptrend = close[i] > sma200_1w_aligned[i]
        weekly_downtrend = close[i] < sma200_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: SMA50 crosses below SMA200 or volatility contraction
            if sma50[i] < sma200[i] or atr[i] < atr[i-1] * 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: SMA50 crosses above SMA200 or volatility contraction
            if sma50[i] > sma200[i] or atr[i] < atr[i-1] * 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            if volume_ok:
                # Golden cross: SMA50 crosses above SMA200 in uptrend
                if weekly_uptrend and sma50[i] > sma200[i] and sma50[i-1] <= sma200[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Death cross: SMA50 crosses below SMA200 in downtrend
                elif weekly_downtrend and sma50[i] < sma200[i] and sma50[i-1] >= sma200[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals