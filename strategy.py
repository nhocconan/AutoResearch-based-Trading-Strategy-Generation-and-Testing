#!/usr/bin/env python3
# 6h_cci_extreme_reversion_v1
# Hypothesis: Mean reversion using CCI(40) on 6h with 1d trend filter.
# Enter long when CCI < -250 (extreme oversold) and price > 1d EMA50.
# Enter short when CCI > +250 (extreme overbought) and price < 1d EMA50.
# Exit when CCI returns to zero or trend filter fails.
# Designed for 15-30 trades/year on 6h to avoid fee drag. Works in bull/bear via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_cci_extreme_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 6h CCI (40-period)
    period = 40
    tp = (high + low + close) / 3
    sma_tp = np.full(n, np.nan)
    mad = np.full(n, np.nan)
    
    for i in range(period, n):
        sma_tp[i] = np.mean(tp[i-period:i])
        md = np.mean(np.abs(tp[i-period:i] - sma_tp[i]))
        mad[i] = md if md > 0 else 1e-10
    
    cci = np.full(n, np.nan)
    for i in range(period, n):
        cci[i] = (tp[i] - sma_tp[i]) / (0.015 * mad[i])
    
    # 1-day EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(50, 40)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(cci[i]) or np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: CCI returns to zero or trend filter fails
            if cci[i] >= 0 or close[i] <= ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI returns to zero or trend filter fails
            if cci[i] <= 0 or close[i] >= ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: extreme oversold with uptrend filter
            if cci[i] < -250 and close[i] > ema50_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: extreme overbought with downtrend filter
            elif cci[i] > 250 and close[i] < ema50_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals