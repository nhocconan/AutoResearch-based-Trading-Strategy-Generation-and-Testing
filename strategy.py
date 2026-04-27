#!/usr/bin/env python3
"""
6h_ElderRay_ZeroLag_MACD_Confluence
Hypothesis: 6h strategy combining Elder Ray (Bull/Bear Power) with ZeroLag MACD and 1w trend filter. 
Elder Ray measures bull/bear power relative to EMA13. ZeroLag MACD reduces lag for timely entries. 
1w EMA50 trend filter ensures alignment with weekly momentum. Designed for BTC/ETH robustness 
in both bull and bear markets via trend filter and momentum confirmation. Targets 50-150 trades 
over 4 years (12-37/year) with 0.25 position size. Uses discrete levels to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate EMA13 for Elder Ray (on 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # ZeroLag MACD: reduces lag by adding the difference between price and EMA
    # EMA calculations
    ema_fast = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema_slow = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema_fast - ema_slow
    # ZeroLag: add (price - EMA) to reduce lag
    zero_lag_macd = macd_line + (close - ema_fast)
    signal_line = pd.Series(zero_lag_macd).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = zero_lag_macd - signal_line
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Fixed position size to minimize churn
    
    # Warmup: need 1w EMA50 (50), EMA13 (13), ZeroLag MACD (max(12,26,9)=26)
    start_idx = max(50, 13, 26)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(zero_lag_macd[i]) or np.isnan(signal_line[i])):
            signals[i] = 0.0
            continue
        
        ema_1w_val = ema_50_1w_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        zl_macd = zero_lag_macd[i]
        zl_signal = signal_line[i]
        
        if position == 0:
            # Look for entry: Elder Ray + ZeroLag MACD confluence with 1w trend filter
            long_condition = (bull_val > 0 and  # Bull power positive
                            zl_macd > zl_signal and  # MACD bullish crossover
                            close[i] > ema_1w_val)   # Above weekly EMA (uptrend)
            
            short_condition = (bear_val < 0 and   # Bear power negative
                             zl_macd < zl_signal and  # MACD bearish crossover
                             close[i] < ema_1w_val)   # Below weekly EMA (downtrend)
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: bear power turns negative OR MACD histogram turns negative
            if bear_val < 0 or zl_macd < zl_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: bull power turns positive OR MACD histogram turns positive
            if bull_val > 0 or zl_macd > zl_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_ZeroLag_MACD_Confluence"
timeframe = "6h"
leverage = 1.0