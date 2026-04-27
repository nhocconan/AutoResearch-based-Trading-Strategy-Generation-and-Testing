#!/usr/bin/env python3
"""
12h_TRIX_12_Signal_Line_Cross_1wTrend_VolumeSpike
Hypothesis: TRIX 12-period crossing its signal line captures momentum shifts. 
Traded only when aligned with weekly trend (price > 1w EMA50 for long, < for short) 
and volume > 2x 20-period average. Uses 12h timeframe to reduce trade frequency 
and fee drag. Discrete sizing (0.25) balances return and risk. 
Targets 50-150 total trades over 4 years (~12-37/year).
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate TRIX (12,9) on close prices
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) then % change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = 100 * (pd.Series(ema3).pct_change().values)
    
    # Signal line: 9-period EMA of TRIX
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    # Align indicators to 12h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    trix_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), trix)
    trix_signal_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), trix_signal)
    volume_confirm_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need EMA50 (50), TRIX calculation (12*3=36 + 9 for signal)
    start_idx = max(50, 36 + 9, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(trix_aligned[i]) or 
            np.isnan(trix_signal_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema50 = ema50_1w_aligned[i]
        trix_val = trix_aligned[i]
        trix_sig = trix_signal_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        # Detect TRIX crossover
        trix_cross_up = trix_val > trix_sig and trix_aligned[i-1] <= trix_signal_aligned[i-1]
        trix_cross_down = trix_val < trix_sig and trix_aligned[i-1] >= trix_signal_aligned[i-1]
        
        if position == 0:
            # Long when TRIX crosses up, price above weekly EMA50, and volume confirmation
            if trix_cross_up and close_val > ema50 and vol_conf:
                signals[i] = size
                position = 1
                entry_price = close_val
            # Short when TRIX crosses down, price below weekly EMA50, and volume confirmation
            elif trix_cross_down and close_val < ema50 and vol_conf:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit long when TRIX crosses down or price closes below weekly EMA50
            if trix_cross_down or close_val < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when TRIX crosses up or price closes above weekly EMA50
            if trix_cross_up or close_val > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_TRIX_12_Signal_Line_Cross_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0