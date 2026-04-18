#!/usr/bin/env python3
"""
1d_WaveTrend_Oscillator_Reversal
Hypothesis: WaveTrend oscillator identifies overbought/oversold conditions on daily timeframe.
Uses WTL1/WTL2 crossovers for entries with weekly trend filter (EMA34) and volume confirmation.
Designed for low-frequency signals (target: 10-25 trades/year) to minimize fee drag in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly trend filter: EMA34
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # WaveTrend parameters
    n1 = 10
    n2 = 21
    
    # Calculate typical price
    tp = (high + low + close) / 3
    
    # EMA of typical price
    ema_tp = pd.Series(tp).ewm(span=n1, adjust=False, min_periods=n1).mean().values
    
    # Absolute deviation
    dev = np.abs(tp - ema_tp)
    
    # EMA of deviation
    ema_dev = pd.Series(dev).ewm(span=n1, adjust=False, min_periods=n1).mean().values
    
    # Avoid division by zero
    epsilon = 1e-10
    ci = (tp - ema_tp) / (0.015 * ema_dev + epsilon)
    
    # TCI smoothed
    tci = pd.Series(ci).ewm(span=n2, adjust=False, min_periods=n2).mean().values
    wt1 = tci
    
    # Signal line (SMA of WT1)
    wt2 = pd.Series(wt1).rolling(window=4, min_periods=4).mean().values
    
    # Volume confirmation: >1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 34)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(wt1[i]) or 
            np.isnan(wt2[i]) or
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        wt1_val = wt1[i]
        wt2_val = wt2[i]
        weekly_trend = ema_34_1w_aligned[i]
        vol_conf = volume_confirmed[i]
        price = close[i]
        
        if position == 0:
            # Long: WT1 crosses above WT2 from oversold (< -60) with weekly uptrend and volume
            if (wt1_val > wt2_val and 
                wt1[i-1] <= wt2[i-1] and 
                wt1_val < -60 and 
                price > weekly_trend and 
                vol_conf):
                signals[i] = 0.25
                position = 1
            # Short: WT1 crosses below WT2 from overbought (> 60) with weekly downtrend and volume
            elif (wt1_val < wt2_val and 
                  wt1[i-1] >= wt2[i-1] and 
                  wt1_val > 60 and 
                  price < weekly_trend and 
                  vol_conf):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: WT1 crosses below WT2 OR weekly trend turns down
            if wt1_val < wt2_val and wt1[i-1] >= wt2[i-1]:
                signals[i] = 0.0
                position = 0
            elif price < weekly_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: WT1 crosses above WT2 OR weekly trend turns up
            if wt1_val > wt2_val and wt1[i-1] <= wt2[i-1]:
                signals[i] = 0.0
                position = 0
            elif price > weekly_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WaveTrend_Oscillator_Reversal"
timeframe = "1d"
leverage = 1.0