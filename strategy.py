#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_TRIX_WeeklyTrend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate TRIX (15-period EMA of EMA of EMA) on weekly close
    close_1w = df_1w['close'].values
    ema1 = pd.Series(close_1w).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = pd.Series(ema3).pct_change(periods=1).values  # TRIX as percentage change
    
    # Align TRIX to daily timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1w, trix)
    
    # Weekly EMA(34) for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike detection: current volume > 2.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for TRIX and EMA calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trix_aligned[i]) or np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trix_val = trix_aligned[i]
        ema_val = ema34_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: TRIX crosses above zero, above weekly EMA, with volume spike
            if (trix_val > 0 and trix_aligned[i-1] <= 0 and 
                close[i] > ema_val and vol_spike):
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses below zero, below weekly EMA, with volume spike
            elif (trix_val < 0 and trix_aligned[i-1] >= 0 and 
                  close[i] < ema_val and vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX crosses below zero OR price drops below weekly EMA
            if (trix_val < 0 and trix_aligned[i-1] >= 0) or close[i] < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX crosses above zero OR price rises above weekly EMA
            if (trix_val > 0 and trix_aligned[i-1] <= 0) or close[i] > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals