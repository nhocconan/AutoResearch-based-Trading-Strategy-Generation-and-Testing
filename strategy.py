#!/usr/bin/env python3
# 1d_Trix_WeeklyTrend_VolumeBreakout
# Hypothesis: Daily TRIX crossover for momentum with weekly EMA trend filter and volume surge confirmation. 
# Works in bull markets via trend-following crossovers and in bear markets via mean-reversion during low-volume pullbacks.
# Volume filter reduces false signals, weekly trend filter ensures alignment with higher timeframe momentum.
# Target: 15-25 trades/year per symbol to minimize fee drag while capturing sustained moves.

timeframe = "1d"
name = "1d_Trix_WeeklyTrend_VolumeBreakout"
leverage = 1.0

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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) == 0:
        return np.zeros(n)
    
    # Calculate weekly EMA40 for trend filter
    ema_40_weekly = pd.Series(df_weekly['close'].values).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_40_weekly)
    
    # Calculate TRIX (15-period) on daily close
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) - 1
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = (ema3 / ema3.shift(1) - 1) * 100
    trix = trix.fillna(0).values
    
    # Calculate TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Volume spike: 2x average volume (20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(45, 20)  # Ensure TRIX and volume MA are valid
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_40_weekly_aligned[i]) or 
            np.isnan(trix[i]) or np.isnan(trix_signal[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above signal line with volume surge and price above weekly EMA40
            if (trix[i] > trix_signal[i] and trix[i-1] <= trix_signal[i-1] and
                volume[i] > 2.0 * vol_ma[i] and
                close[i] > ema_40_weekly_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below signal line with volume surge and price below weekly EMA40
            elif (trix[i] < trix_signal[i] and trix[i-1] >= trix_signal[i-1] and
                  volume[i] > 2.0 * vol_ma[i] and
                  close[i] < ema_40_weekly_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX crosses below signal line or trend failure (price below weekly EMA40)
            if (trix[i] < trix_signal[i] and trix[i-1] >= trix_signal[i-1]) or \
               close[i] < ema_40_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX crosses above signal line or trend failure (price above weekly EMA40)
            if (trix[i] > trix_signal[i] and trix[i-1] <= trix_signal[i-1]) or \
               close[i] > ema_40_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals