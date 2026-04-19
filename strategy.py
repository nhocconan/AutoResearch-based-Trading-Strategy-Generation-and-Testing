#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy combining 12-hour momentum (ROC25) with volume confirmation and trend filter from 1-day EMA34.
# Uses 12h ROC25 for momentum signal, confirmed by volume spike, and filtered by daily EMA34 trend.
# Designed to work in both bull and bear markets by requiring alignment between daily trend
# and 12h momentum direction, reducing false signals in choppy conditions.
# Target: 15-25 trades/year per symbol with disciplined entries.
name = "6h_EMA34_1d_ROC25_12h_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily EMA34 for trend bias
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 12-hour ROC25 for momentum
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 25:
        return np.zeros(n)
    
    roc_25_12h = pd.Series(df_12h['close']).pct_change(periods=25).values
    roc_25_12h_aligned = align_htf_to_ltf(prices, df_12h, roc_25_12h)
    
    # Volume spike: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(roc_25_12h_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: positive ROC25 from 12h, above daily EMA34, with volume spike
            if (roc_25_12h_aligned[i] > 0 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: negative ROC25 from 12h, below daily EMA34, with volume spike
            elif (roc_25_12h_aligned[i] < 0 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if ROC25 turns negative or price breaks below daily EMA34
            if (roc_25_12h_aligned[i] < 0) or (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if ROC25 turns positive or price breaks above daily EMA34
            if (roc_25_12h_aligned[i] > 0) or (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals