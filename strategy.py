#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_trix_volume_surge_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for TRIX calculation and volume context
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 18:  # Need enough for EMA(12) triple
        return np.zeros(n)
    
    # 12h TRIX calculation (triple EMA)
    close_12h = df_12h['close'].values
    ema1 = pd.Series(close_12h).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean()
    trix = 100 * (ema3 / ema3.shift(1) - 1)
    trix = trix.fillna(0).values
    
    # 12h TRIX signal line (EMA of TRIX)
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # 12h volume average for surge detection
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 12h indicators to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_12h, trix)
    trix_signal_aligned = align_htf_to_ltf(prices, df_12h, trix_signal)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Volume surge: current 4h volume > 1.5 * 12h volume MA
    volume_surge = volume > (1.5 * vol_ma_12h_aligned)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(trix_signal_aligned[i]) or 
            np.isnan(volume_surge[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: TRIX crosses above signal line with volume surge
        long_signal = (trix_aligned[i] > trix_signal_aligned[i] and 
                      trix_aligned[i-1] <= trix_signal_aligned[i-1] and
                      volume_surge[i])
        
        # Short: TRIX crosses below signal line with volume surge
        short_signal = (trix_aligned[i] < trix_signal_aligned[i] and 
                       trix_aligned[i-1] >= trix_signal_aligned[i-1] and
                       volume_surge[i])
        
        # Exit: TRIX crosses back through signal line
        exit_long = (position == 1 and trix_aligned[i] < trix_signal_aligned[i])
        exit_short = (position == -1 and trix_aligned[i] > trix_signal_aligned[i])
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals