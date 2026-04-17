#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA trend filter and ATR-based volatility filter.
Long when Bull Power > 0 AND 12h EMA34 > 12h EMA89 (uptrend) AND ATR(14) < 1.5x ATR(50) (low volatility).
Short when Bear Power < 0 AND 12h EMA34 < 12h EMA89 (downtrend) AND ATR(14) < 1.5x ATR(50) (low volatility).
Exit when trend reverses (EMA cross) or volatility expands (ATR(14) > 2x ATR(50)).
Uses 12h for trend and volatility filters, 6h for Elder Ray calculation and execution.
Designed to capture trending moves with low volatility pullbacks in both bull and bear markets. Target: 12-30 trades/year per symbol.
"""

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
    
    # Get 12h data for trend and volatility filters
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMAs for trend filter
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_12h = pd.Series(close_12h).ewm(span=89, adjust=False, min_periods=89).mean().values
    uptrend_12h = ema34_12h > ema89_12h
    downtrend_12h = ema34_12h < ema89_12h
    
    # Calculate 12h ATR for volatility filter
    tr_12h = np.maximum(high_12h - low_12h, 
                        np.absolute(high_12h - np.roll(close_12h, 1)),
                        np.absolute(low_12h - np.roll(close_12h, 1)))
    tr_12h[0] = high_12h[0] - low_12h[0]
    atr14_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    atr50_12h = pd.Series(tr_12h).rolling(window=50, min_periods=50).mean().values
    low_vol_12h = atr14_12h < (1.5 * atr50_12h)
    high_vol_12h = atr14_12h > (2.0 * atr50_12h)
    
    # Calculate 6h Elder Ray (Bull/Bear Power)
    ema13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13_6h  # Bull Power = High - EMA13
    bear_power = low - ema13_6h   # Bear Power = Low - EMA13
    
    # Align all 12h indicators to 6h timeframe
    uptrend_aligned = align_htf_to_ltf(prices, df_12h, uptrend_12h)
    downtrend_aligned = align_htf_to_ltf(prices, df_12h, downtrend_12h)
    low_vol_aligned = align_htf_to_ltf(prices, df_12h, low_vol_12h)
    high_vol_aligned = align_htf_to_ltf(prices, df_12h, high_vol_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(uptrend_aligned[i]) or 
            np.isnan(downtrend_aligned[i]) or
            np.isnan(low_vol_aligned[i]) or
            np.isnan(high_vol_aligned[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_entry = (bull_power[i] > 0 and 
                     uptrend_aligned[i] and 
                     low_vol_aligned[i])
        short_entry = (bear_power[i] < 0 and 
                      downtrend_aligned[i] and 
                      low_vol_aligned[i])
        
        # Exit conditions
        exit_long = (not uptrend_aligned[i]) or high_vol_aligned[i]
        exit_short = (not downtrend_aligned[i]) or high_vol_aligned[i]
        
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_EMATrend_ATRVol_Filter"
timeframe = "6h"
leverage = 1.0