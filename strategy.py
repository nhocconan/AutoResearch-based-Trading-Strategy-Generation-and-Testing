#!/usr/bin/env python3
# 1d_trix_volume_regime
# Hypothesis: On 1d timeframe, use TRIX (15) with volume confirmation and weekly trend filter to capture momentum in both bull and bear markets.
# Long when TRIX crosses above zero with volume > 1.5x average and weekly uptrend.
# Short when TRIX crosses below zero with volume > 1.5x average and weekly downtrend.
# Exit when TRIX crosses back across zero.
# Target: 20-50 trades over 4 years (5-12/year) to minimize fee drag while capturing strong momentum moves.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_trix_volume_regime"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate TRIX (15) - triple smoothed EMA of ROC
    # TRIX = EMA(EMA(EMA(ROC, 15), 15), 15)
    roc = np.zeros_like(close)
    roc[1:] = (close[1:] - close[:-1]) / close[:-1] * 100  # ROC in percentage
    
    # Triple EMA smoothing
    ema1 = pd.Series(roc).ewm(span=15, min_periods=15, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, min_periods=15, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, min_periods=15, adjust=False).mean().values
    trix = ema3
    
    # Volume confirmation: 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Weekly trend filter: EMA50 on weekly timeframe
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    weekly_close = df_weekly['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(trix[i]) or np.isnan(trix[i-1]) or np.isnan(avg_volume[i]) or np.isnan(weekly_ema50_aligned[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: TRIX crosses below zero
            if trix[i] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TRIX crosses above zero
            if trix[i] >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Weekly trend filter
            weekly_uptrend = close[i] > weekly_ema50_aligned[i]
            weekly_downtrend = close[i] < weekly_ema50_aligned[i]
            
            # TRIX zero cross
            trix_cross_up = trix[i-1] <= 0 and trix[i] > 0
            trix_cross_down = trix[i-1] >= 0 and trix[i] < 0
            
            # Long entry: TRIX crosses above zero with volume and weekly uptrend
            if trix_cross_up and volume_ok and weekly_uptrend:
                position = 1
                signals[i] = 0.25
            # Short entry: TRIX crosses below zero with volume and weekly downtrend
            elif trix_cross_down and volume_ok and weekly_downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals