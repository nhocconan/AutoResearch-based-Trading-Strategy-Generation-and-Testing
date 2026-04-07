#!/usr/bin/env python3
"""
1d_trix_1w_trend_volume_v1
Hypothesis: On daily timeframe, use TRIX (15) momentum with weekly EMA40 trend filter and volume confirmation.
Enter long when TRIX crosses above zero in weekly uptrend with volume > 1.3x average, short when TRIX crosses below zero in weekly downtrend with volume > 1.3x average.
Exit on opposite TRIX cross. TRIX filters noise and captures momentum shifts; weekly trend ensures alignment with higher timeframe direction.
Designed for low frequency (7-25 trades/year) to minimize fee drain.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_trix_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Calculate weekly EMA40 for trend
    weekly_close = df_1w['close'].values
    weekly_ema40 = pd.Series(weekly_close).ewm(span=40, adjust=False, min_periods=40).mean().values
    weekly_ema40_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema40)
    
    # Calculate TRIX(15) on daily close
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) - 1 period ago, then percent change
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1) * 100
    trix[0] = 0  # first value has no previous
    
    # Calculate 20-period average volume for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(40, n):  # Start after EMA40 warmup
        # Skip if weekly data not available
        if np.isnan(weekly_ema40_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine weekly trend
        weekly_uptrend = weekly_ema40_aligned[i] > np.roll(weekly_ema40_aligned, 1)[i] if i > 0 else False
        weekly_downtrend = weekly_ema40_aligned[i] < np.roll(weekly_ema40_aligned, 1)[i] if i > 0 else False
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        if position == 1:  # Long position
            # Exit when TRIX crosses below zero
            if trix[i] < 0 and trix[i-1] >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when TRIX crosses above zero
            if trix[i] > 0 and trix[i-1] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: TRIX crosses above zero in weekly uptrend with volume confirmation
            long_entry = (trix[i] > 0 and trix[i-1] <= 0) and weekly_uptrend and vol_confirm
            # Short entry: TRIX crosses below zero in weekly downtrend with volume confirmation
            short_entry = (trix[i] < 0 and trix[i-1] >= 0) and weekly_downtrend and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals