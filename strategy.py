#!/usr/bin/env python3
"""
#101002 - 12h_ChaikinVolatility_Breakout_VolumeTrend
Hypothesis: Chaikin Volatility (10-period high-low range expansion) breakout on 12h timeframe with volume confirmation and 1d EMA trend filter.
Works in both bull and bear markets by trading volatility breakouts in direction of higher timeframe trend. Uses Chaikin Volatility to detect expansion phases and volume to confirm institutional participation.
Target: 15-30 trades/year to minimize fee drag. Discrete position sizing (0.25) to reduce churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Chaikin Volatility calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Chaikin Volatility: 10-period EMA of (High - Low)
    hl_range = high_12h - low_12h
    cv = pd.Series(hl_range).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate CV rate of change (1-period) for breakout detection
    cv_roc = np.diff(cv, prepend=cv[0])
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    # Align 12h indicators to lower timeframe
    cv_aligned = align_htf_to_ltf(prices, df_12h, cv)
    cv_roc_aligned = align_htf_to_ltf(prices, df_12h, cv_roc)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(cv_aligned[i]) or np.isnan(cv_roc_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: CV rising (volatility expanding), price above EMA50, volume surge
        if (cv_roc_aligned[i] > 0 and 
            close[i] > ema50_1d_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: CV rising (volatility expanding), price below EMA50, volume surge
        elif (cv_roc_aligned[i] > 0 and 
              close[i] < ema50_1d_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit when volatility contracts (CV ROC turns negative) or opposite condition met
        elif position == 1 and (cv_roc_aligned[i] < 0 or close[i] < ema50_1d_aligned[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (cv_roc_aligned[i] < 0 or close[i] > ema50_1d_aligned[i]):
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

name = "12h_ChaikinVolatility_Breakout_VolumeTrend"
timeframe = "12h"
leverage = 1.0