#!/usr/bin/env python3
name = "4h_TRIX_0_Cross_1dTrend_Volume"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # TRIX(12) calculation (triple EMA of 1-period ROC)
    roc = np.diff(np.log(close), prepend=np.log(close[0])) * 100
    ema1 = pd.Series(roc).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = ema3
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike detection (4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 34)
    
    for i in range(start_idx, n):
        if (np.isnan(trix[i]) or np.isnan(trix[i-1]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # TRIX zero cross signals
        trix_cross_up = trix[i-1] <= 0 and trix[i] > 0
        trix_cross_down = trix[i-1] >= 0 and trix[i] < 0
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: TRIX crosses above 0 in daily uptrend with volume
            if trix_cross_up and ema34_1d_aligned[i] > ema34_1d_aligned[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below 0 in daily downtrend with volume
            elif trix_cross_down and ema34_1d_aligned[i] < ema34_1d_aligned[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX crosses below 0 or trend breaks
            if trix_cross_down or ema34_1d_aligned[i] <= ema34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX crosses above 0 or trend breaks
            if trix_cross_up or ema34_1d_aligned[i] >= ema34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h TRIX zero cross with daily trend filter and volume confirmation
# - TRIX (Triple Exponential Average) measures momentum of momentum
# - Zero cross indicates change in momentum direction
# - Daily EMA34 trend filter ensures alignment with higher timeframe trend
# - Volume confirmation (1.5x average) reduces false signals
# - Works in both bull (zero crosses up in uptrend) and bear (zero crosses down in downtrend)
# - Position size 0.25 targets ~20-40 trades/year to avoid fee drag
# - TRIX is less noisy than MACD and provides clearer signals
# - TRIX zero cross + trend + volume combination not recently tried on 4h
# - Aims for 80-160 total trades over 4 years (20-40/year) to stay within limits
# - Effective in ranging markets as TRIX oscillates around zero
# - Trend filter prevents counter-trend trading in strong trends
# - Volume adds confirmation to reduce whipsaws
# - Exit on signal reversal or trend breakdown provides clear risk control
# - Simple 3-condition logic minimizes overfitting and curve-fitting risks