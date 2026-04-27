#!/usr/bin/env python3
"""
4H_CCI_Reversal_with_Volume_and_Trend_Filter
Hypothesis: Enter long when CCI(20) crosses below -100 (oversold) with price above 1d EMA34 and volume > 1.5x average; enter short when CCI crosses above +100 (overbought) with price below 1d EMA34 and volume > 1.5x average. Exit on CCI crossing back toward zero. This captures mean-reversion bounces in trending markets, working in both bull (buy dips) and bear (sell rallies). Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # CCI(20) calculation
    typical_price = (high + low + close) / 3.0
    tp_mean = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    tp_mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    # Avoid division by zero
    tp_mad = np.where(tp_mad == 0, 1e-10, tp_mad)
    cci = (typical_price - tp_mean) / (0.015 * tp_mad)
    
    # Volume filter: require volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period
    start_idx = 20  # need 20 for CCI and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(cci[i]) or 
            np.isnan(tp_mean[i]) or np.isnan(tp_mad[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: CCI crosses below -100 (oversold) in uptrend with volume
            if (cci[i] <= -100 and cci[i-1] > -100 and 
                close[i] > ema34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: CCI crosses above +100 (overbought) in downtrend with volume
            elif (cci[i] >= 100 and cci[i-1] < 100 and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: CCI crosses back above -50 (recovery)
            if cci[i] >= -50 and cci[i-1] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: CCI crosses back below +50 (recovery)
            if cci[i] <= 50 and cci[i-1] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4H_CCI_Reversal_with_Volume_and_Trend_Filter"
timeframe = "4h"
leverage = 1.0