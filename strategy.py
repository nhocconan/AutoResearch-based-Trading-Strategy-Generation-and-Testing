#!/usr/bin/env python3
"""
6h_1w_1d_adaptive_cci_v1
Hypothesis: 6-hour strategy using CCI on weekly and daily timeframes with adaptive thresholds based on volatility regime.
Long when weekly CCI > +100 and daily CCI crosses above +50; short when weekly CCI < -100 and daily CCI crosses below -50.
Uses ATR-based volatility filter to avoid chop and dynamic position sizing. Designed to catch strong trends while avoiding range-bound markets.
Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag.
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
    
    # Get weekly data for primary trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly CCI(20)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Typical price
    tp_1w = (high_1w + low_1w + close_1w) / 3
    # Moving average of typical price
    ma_tp_1w = pd.Series(tp_1w).rolling(window=20, min_periods=20).mean().values
    # Mean deviation
    md_1w = pd.Series(tp_1w).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    # CCI calculation
    cci_1w = (tp_1w - ma_tp_1w) / (0.015 * md_1w)
    cci_1w = np.where(md_1w == 0, 0, cci_1w)  # avoid division by zero
    cci_1w_aligned = align_htf_to_ltf(prices, df_1w, cci_1w)
    
    # Get daily data for entry signal and volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily CCI(14) for entry timing
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tp_1d = (high_1d + low_1d + close_1d) / 3
    ma_tp_1d = pd.Series(tp_1d).rolling(window=14, min_periods=14).mean().values
    md_1d = pd.Series(tp_1d).rolling(window=14, min_periods=14).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci_1d = (tp_1d - ma_tp_1d) / (0.015 * md_1d)
    cci_1d = np.where(md_1d == 0, 0, cci_1d)
    cci_1d_aligned = align_htf_to_ltf(prices, df_1d, cci_1d)
    
    # Calculate ATR for volatility filter and position sizing
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - close)
    tr3 = np.abs(np.roll(low, 1) - close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(cci_1w_aligned[i]) or np.isnan(cci_1d_aligned[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely low volatility (chop) and extreme volatility (chaos)
        if i >= 20:
            atr_ma = np.mean(atr[max(0, i-20):i])
            vol_ratio = atr[i] / atr_ma if atr_ma > 0 else 1.0
            volatility_filter = 0.5 <= vol_ratio <= 2.5  # only trade in normal volatility regimes
        else:
            volatility_filter = True
        
        # Entry conditions: weekly trend + daily momentum
        long_entry = (cci_1w_aligned[i] > 100 and 
                      cci_1d_aligned[i] > 50 and 
                      i > 20 and cci_1d_aligned[i-1] <= 50 and  # daily CCI crosses above 50
                      volatility_filter)
        short_entry = (cci_1w_aligned[i] < -100 and 
                       cci_1d_aligned[i] < -50 and 
                       i > 20 and cci_1d_aligned[i-1] >= -50 and  # daily CCI crosses below -50
                       volatility_filter)
        
        # Exit conditions: weekly trend reversal or daily mean reversion
        long_exit = (cci_1w_aligned[i] < 0 or 
                     cci_1d_aligned[i] < -50 or 
                     (i > 20 and cci_1d_aligned[i] < cci_1d_aligned[i-1] and cci_1d_aligned[i] < 0))
        short_exit = (cci_1w_aligned[i] > 0 or 
                      cci_1d_aligned[i] > 50 or 
                      (i > 20 and cci_1d_aligned[i] > cci_1d_aligned[i-1] and cci_1d_aligned[i] > 0))
        
        # Dynamic position sizing based on volatility (inverse vol)
        if i >= 20:
            atr_ma = np.mean(atr[max(0, i-20):i])
            vol_scaling = np.clip(atr_ma / atr[i], 0.5, 2.0)  # inverse vol: smaller size in high vol
        else:
            vol_scaling = 1.0
        base_size = 0.25
        position_size = base_size * vol_scaling
        position_size = np.clip(position_size, 0.20, 0.30)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_1d_adaptive_cci_v1"
timeframe = "6h"
leverage = 1.0