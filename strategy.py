#!/usr/bin/env python3
# 4H_1D_Trix_Volume_Trend
# Hypothesis: On 4h timeframe, enter long when TRIX(12) crosses above zero with 1d uptrend and volume confirmation.
# Short when TRIX(12) crosses below zero with 1d downtrend and volume confirmation.
# Uses TRIX for momentum reversal detection, 1d trend filter to avoid counter-trend trades, and volume to confirm strength.
# Target: 20-50 trades/year per symbol (80-200 total over 4 years).

name = "4H_1D_Trix_Volume_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for TRIX and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate TRIX(12) on 1d close: triple EMA then % change
    ema1 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix_raw = 100 * (ema3 / ema3.shift(1) - 1)
    trix = trix_raw.fillna(0).values
    
    # 1d trend: EMA(34) on close
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = close_1d > ema_34
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    # Align 1d indicators to 4h
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(trix_aligned[i]) or np.isnan(trend_up_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: TRIX crosses above zero + 1d uptrend + volume confirmation
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and trend_up_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: TRIX crosses below zero + 1d downtrend + volume confirmation
            elif trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and not trend_up_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: TRIX crosses below zero or trend changes
            if trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 or not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: TRIX crosses above zero or trend changes
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 or trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals