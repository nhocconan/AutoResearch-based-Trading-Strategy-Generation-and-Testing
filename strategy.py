#!/usr/bin/env python3
"""
6h_camarilla_pivot_1w_trend_volume_v1
Hypothesis: Weekly pivot levels act as strong support/resistance on 6h timeframe.
Price reversals at weekly Camarilla levels with volume confirmation and trend alignment
capture both mean reversion and breakout moves. Weekly trend filter ensures trading
in direction of higher timeframe momentum, working in both bull and bear markets.
Targets 12-37 trades/year with disciplined entries to avoid overtrading.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1w_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_6h = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1-week OHLC for Camarilla pivot calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels for each week
    camarilla_range = 1.1 * (high_1w - low_1w) / 12
    camarilla_s1 = close_1w - camarilla_range  # Support level 1
    camarilla_r1 = close_1w + camarilla_range  # Resistance level 1
    
    # Align Camarilla levels to 6h timeframe
    camarilla_s1_6h = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    camarilla_r1_6h = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    
    # 20-period SMA for volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema50_6h[i]) or 
            np.isnan(camarilla_s1_6h[i]) or 
            np.isnan(camarilla_r1_6h[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches resistance OR trend turns down
            if close[i] >= camarilla_r1_6h[i] or close[i] < ema50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price reaches support OR trend turns up
            if close[i] <= camarilla_s1_6h[i] or close[i] > ema50_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price touches support + volume confirmation + uptrend
            if (close[i] <= camarilla_s1_6h[i] and 
                vol_confirm and 
                close[i] > ema50_6h[i]):
                position = 1
                signals[i] = 0.25
            # Short: price touches resistance + volume confirmation + downtrend
            elif (close[i] >= camarilla_r1_6h[i] and 
                  vol_confirm and 
                  close[i] < ema50_6h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals