#!/usr/bin/env python3
"""
4h_TRIX_Volume_Spike_Momentum
Hypothesis: TRIX (triple EMA crossover) captures momentum shifts, while volume spikes confirm institutional participation.
In bull markets (price above 1-day EMA50), go long on TRIX cross above zero with volume spike.
In bear markets (price below 1-day EMA50), go short on TRIX cross below zero with volume spike.
Uses 4h timeframe for execution with 1d trend filter. Target: 30-50 trades per year (~120-200 over 4 years) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_TRIX_Volume_Spike_Momentum"
timeframe = "4h"
leverage = 1.0

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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1-day EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # TRIX on 4h close: triple EMA of percent change
    # TRIX = EMA(EMA(EMA(ROC, 12), 12), 12) * 100
    roc = np.diff(np.log(close), prepend=np.log(close[0]))  # log return approximation
    ema1 = pd.Series(roc).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = ema3 * 100  # scale for readability
    
    # TRIX signal line (9-period EMA of TRIX)
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # TRIX histogram (main signal: TRIX - signal)
    trix_hist = trix - trix_signal
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need sufficient warmup for EMA calculations
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(trix_hist[i]) or 
            np.isnan(trix_hist[i-1]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market regime from 1-day EMA50
        uptrend_regime = close[i] > ema_50_1d_aligned[i]
        downtrend_regime = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation: volume > 1.5x average
        volume_confirm = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: TRIX histogram crosses above zero in uptrend regime + volume spike
            long_entry = (trix_hist[i] > 0) and (trix_hist[i-1] <= 0) and uptrend_regime and volume_confirm
            # Short: TRIX histogram crosses below zero in downtrend regime + volume spike
            short_entry = (trix_hist[i] < 0) and (trix_hist[i-1] >= 0) and downtrend_regime and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX histogram crosses below zero or regime changes to downtrend
            if (trix_hist[i] < 0) or (not uptrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX histogram crosses above zero or regime changes to uptrend
            if (trix_hist[i] > 0) or (not downtrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals