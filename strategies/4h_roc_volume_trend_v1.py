#!/usr/bin/env python3
# 4h_roc_volume_trend_v1
# Hypothesis: On 4h timeframe, buy when Rate of Change (ROC) crosses above 0 with volume confirmation and 1d trend up,
# sell when ROC crosses below 0 with volume confirmation and 1d trend down. Uses ROC(12) for momentum,
# volume > 1.5x 20-period average for confirmation, and 1d EMA(50) for trend filter.
# Designed to capture momentum shifts in both bull and bear markets with strict entry conditions to limit trades.
# Target: 20-40 trades/year to avoid fee decay while capturing sustained moves.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_roc_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (calculate once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ROC(12) on 4h close
    roc = np.zeros_like(close)
    for i in range(12, n):
        if close[i-12] != 0:
            roc[i] = (close[i] - close[i-12]) / close[i-12] * 100
    
    # Volume confirmation: 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(roc[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ROC crosses below 0 or opposite signal with volume
            if roc[i] < 0 or (roc[i] > 0 and volume[i] > 1.5 * avg_volume[i] and close[i] < ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ROC crosses above 0 or opposite signal with volume
            if roc[i] > 0 or (roc[i] < 0 and volume[i] > 1.5 * avg_volume[i] and close[i] > ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Long entry: ROC crosses above 0 with volume and 1d uptrend
            if roc[i] > 0 and roc[i-1] <= 0 and volume_ok and close[i] > ema_50_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: ROC crosses below 0 with volume and 1d downtrend
            elif roc[i] < 0 and roc[i-1] >= 0 and volume_ok and close[i] < ema_50_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals