#!/usr/bin/env python3
"""
12h_Trix_ZeroCross_1wTrend_Volume_Spike
Hypothesis: TRIX crossing zero on 12h, filtered by 1w EMA trend and volume spikes, to capture medium-term momentum in BTC/ETH. TRIX is sensitive to trend changes and works well with volume confirmation to avoid false signals. Uses volume spike (current > 2.0 * 24-period average) for confirmation. Trend filter uses 1w EMA34 to ensure alignment with weekly momentum. Designed for fewer trades (~20-40/year) to minimize fee drag on 12h timeframe. Works in bull markets via zero-cross longs and bear via zero-cross shorts.
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
    
    # Calculate TRIX on 12h data (triple EMA of ROC)
    # TRIX = EMA(EMA(EMA(ROC, period), period), period)
    roc = np.diff(np.log(close), prepend=np.log(close[0]))  # approximate ROC
    ema1 = pd.Series(roc).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = ema3 * 100  # scale for readability
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: current volume > 2.0 * 24-period average (on 12h data, ~12 days)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for TRIX calculation and volume average
    start_idx = max(45, 24)  # 45 for triple EMA stability
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data not ready
        if (np.isnan(trix[i]) or np.isnan(trix[i-1]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        trix_now = trix[i]
        trix_prev = trix[i-1]
        ema_34_val = ema_34_1w_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Require minimum 24 bars since last exit to avoid churn (~12 days on 12h)
            if bars_since_exit >= 24:
                # Long: TRIX crosses above zero with volume confirmation AND above 1w EMA34 (uptrend)
                if trix_prev <= 0 and trix_now > 0 and vol_conf and close[i] > ema_34_val:
                    signals[i] = size
                    position = 1
                    bars_since_exit = 0
                # Short: TRIX crosses below zero with volume confirmation AND below 1w EMA34 (downtrend)
                elif trix_prev >= 0 and trix_now < 0 and vol_conf and close[i] < ema_34_val:
                    signals[i] = -size
                    position = -1
                    bars_since_exit = 0
        elif position == 1:
            # Exit long: TRIX crosses below zero
            if trix_prev >= 0 and trix_now < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: TRIX crosses above zero
            if trix_prev <= 0 and trix_now > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Trix_ZeroCross_1wTrend_Volume_Spike"
timeframe = "12h"
leverage = 1.0