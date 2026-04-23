#!/usr/bin/env python3
"""
Hypothesis: 4h TRIX(12) crossover with 12h EMA50 trend filter and volume spike confirmation.
Long when TRIX crosses above signal line AND 12h EMA50 rising AND volume > 1.5x 20-period MA.
Short when TRIX crosses below signal line AND 12h EMA50 falling AND volume > 1.5x 20-period MA.
Exit on opposite TRIX crossover or EMA50 reversal.
Uses TRIX for momentum, 12h EMA50 for major trend filter, volume spike for confirmation.
Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
Works in bull (trend filters) and bear (volume spikes on breakdowns).
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
    
    # Calculate TRIX(12) - Triple Exponential Average
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12)
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = pd.Series(ema3).pct_change() * 100  # Percentage change
    trix_values = trix.values
    
    # TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix_values).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Calculate 12h EMA50 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(12+12+12+8, 50, 20)  # TRIX calculation, EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix_values[i]) or np.isnan(trix_signal[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        trix_val = trix_values[i]
        trix_sig = trix_signal[i]
        ema_val = ema_50_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Calculate TRIX crossover
        if i >= start_idx + 1:
            trix_prev = trix_values[i-1]
            trix_sig_prev = trix_signal[i-1]
            trix_cross_up = trix_prev <= trix_sig_prev and trix_val > trix_sig
            trix_cross_down = trix_prev >= trix_sig_prev and trix_val < trix_sig
            
            # Calculate EMA50 slope for trend direction
            ema_prev = ema_50_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            trix_cross_up = False
            trix_cross_down = False
            ema_rising = False
            ema_falling = False
        
        # Volume filter: volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: TRIX crosses above signal AND EMA50 rising AND volume filter
            if trix_cross_up and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below signal AND EMA50 falling AND volume filter
            elif trix_cross_down and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: TRIX crosses below signal OR EMA50 starts falling
                if trix_cross_down or (i >= start_idx + 1 and ema_val < ema_50_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: TRIX crosses above signal OR EMA50 starts rising
                if trix_cross_up or (i >= start_idx + 1 and ema_val > ema_50_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_TRIX12_Crossover_12hEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0