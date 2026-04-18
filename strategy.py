#!/usr/bin/env python3
"""
12h_Trix_15_Signal_MeanReversion
Hypothesis: TRIX (15) mean reversion on 12h timeframe with volume confirmation and weekly trend filter.
TRIX crossing above zero with volume spike indicates bullish momentum; crossing below zero indicates bearish.
Weekly EMA34 filter ensures alignment with higher timeframe trend.
Target: 12-37 trades/year to stay within optimal range for 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # TRIX (15) calculation: triple EMA of ROC
    roc = np.diff(np.log(close), prepend=np.log(close[0])) * 100  # ROC as percentage
    ema1 = pd.Series(roc).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = ema3  # TRIX is the third EMA of ROC
    
    # Weekly trend filter: EMA34
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike: >2.0x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(45, 30)  # Warmup for TRIX and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(trix[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        trix_val = trix[i]
        trix_prev = trix[i-1] if i > 0 else 0
        ema34 = ema_34_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: TRIX crosses above zero with volume spike and weekly uptrend
            if trix_val > 0 and trix_prev <= 0 and vol_spike and close[i] > ema34:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero with volume spike and weekly downtrend
            elif trix_val < 0 and trix_prev >= 0 and vol_spike and close[i] < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: TRIX crosses below zero OR weekly trend turns down
            if trix_val < 0 and trix_prev >= 0:
                signals[i] = 0.0
                position = 0
            elif close[i] < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: TRIX crosses above zero OR weekly trend turns up
            if trix_val > 0 and trix_prev <= 0:
                signals[i] = 0.0
                position = 0
            elif close[i] > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Trix_15_Signal_MeanReversion"
timeframe = "12h"
leverage = 1.0