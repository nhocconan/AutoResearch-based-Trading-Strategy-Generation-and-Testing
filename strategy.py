#!/usr/bin/env python3
"""
1d_WeeklyEMA_Cross_with_Volume_Spike_and_Trend_v1
Hypothesis: Buy when price crosses above 50-week EMA with volume spike; sell when price crosses below 50-week EMA with volume spike. Use daily timeframe for execution and weekly EMA for trend filter. Volume spike confirms institutional participation. Designed for low trade frequency (<25/year) to minimize fee drag while capturing major trend changes in both bull and bear markets.
"""

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
    
    # Volume spike: >2.0x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Weekly 50 EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 40  # Need volume MA and weekly EMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_50_val = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price crosses above weekly 50 EMA with volume spike
            if price > ema_50_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below weekly 50 EMA with volume spike
            elif price < ema_50_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price crosses below weekly 50 EMA
            if price < ema_50_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price crosses above weekly 50 EMA
            if price > ema_50_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WeeklyEMA_Cross_with_Volume_Spike_and_Trend_v1"
timeframe = "1d"
leverage = 1.0