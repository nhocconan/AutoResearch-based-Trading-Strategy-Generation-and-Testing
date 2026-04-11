#!/usr/bin/env python3
"""
4h_1d_cci_reversal_v1
Strategy: 4h CCI reversal with 1d trend filter
Timeframe: 4h
Leverage: 1.0
Hypothesis: Uses CCI(20) extremes on 4h (>100 short, <-100 long) combined with 1d EMA50 trend filter. Enters counter-trend reversals only when aligned with higher timeframe trend to avoid counter-trend trades in strong moves. Designed to work in both bull/bear markets by following 1d trend. Target: 80-120 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_cci_reversal_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h CCI(20)
    tp = (high + low + close) / 3.0
    ma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    md = pd.Series(np.abs(tp - ma_tp)).rolling(window=20, min_periods=20).mean().values
    cci = (tp - ma_tp) / (0.015 * md)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(cci[i]) or np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        
        # CCI extreme conditions
        cci_overbought = cci[i] > 100
        cci_oversold = cci[i] < -100
        
        # Long: CCI oversold in uptrend
        long_signal = cci_oversold and uptrend_1d
        
        # Short: CCI overbought in downtrend
        short_signal = cci_overbought and downtrend_1d
        
        # Exit when CCI returns to neutral zone
        exit_long = position == 1 and cci[i] > -50
        exit_short = position == -1 and cci[i] < 50
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Uses CCI(20) extremes on 4h (>100 short, <-100 long) combined with 1d EMA50 trend filter. Enters counter-trend reversals only when aligned with higher timeframe trend to avoid counter-trend trades in strong moves. Designed to work in both bull/bear markets by following 1d trend. Target: 80-120 total trades over 4 years.