#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_VWAP_Deviation_1dTrend_VolumeFilter"
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
    
    # Calculate VWAP for 6h window
    typical_price = (high + low + close) / 3.0
    vwap_numerator = typical_price * volume
    vwap_denominator = volume
    vwap = pd.Series(vwap_numerator).rolling(window=4, min_periods=4).sum().values / \
           pd.Series(vwap_denominator).rolling(window=4, min_periods=4).sum().values
    
    # Calculate deviation from VWAP as percentage
    vwap_dev = (close - vwap) / vwap * 100.0
    
    # 1d data for trend filter and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d volume average for context
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap_dev[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_avg_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price significantly below VWAP (mean reversion) AND above 1d EMA50 (uptrend)
            long_cond = (vwap_dev[i] < -1.5 and 
                        close[i] > ema50_1d_aligned[i])
            
            # Short: Price significantly above VWAP (mean reversion) AND below 1d EMA50 (downtrend)
            short_cond = (vwap_dev[i] > 1.5 and 
                         close[i] < ema50_1d_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price returns to VWAP OR breaks below 1d EMA50
            if vwap_dev[i] > -0.5 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price returns to VWAP OR breaks above 1d EMA50
            if vwap_dev[i] < 0.5 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Mean reversion from 6h VWAP with 1d EMA50 trend filter.
# In uptrends (price > 1d EMA50), buy when price deviates significantly below VWAP.
# In downtrends (price < 1d EMA50), sell when price deviates significantly above VWAP.
# VWAP deviation >1.5% provides entry signal, reversion to within 0.5% provides exit.
# Works in both bull and bear markets by aligning with higher timeframe trend.
# 6h timeframe targets 12-37 trades/year to minimize fee drag. Discrete sizing (0.25) reduces churn.