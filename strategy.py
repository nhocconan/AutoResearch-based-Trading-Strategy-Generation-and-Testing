#!/usr/bin/env python3
"""
1d_1w_cci_trend_reversal
Hypothesis: 1-day strategy using weekly CCI for trend identification and daily CCI for mean-reversion entries.
In bull markets, buy dips in uptrend; in bear markets, sell rallies in downtrend.
Uses weekly CCI(20) > 100 for uptrend, < -100 for downtrend, and daily CCI(14) for entry/exit.
Designed to work in both bull and bear markets by aligning with higher timeframe trend.
Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.
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
    
    # Get weekly data for trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly CCI(20) for trend
    tp_1w = (high_1w + low_1w + close_1w) / 3.0
    sma_tp_1w = pd.Series(tp_1w).rolling(window=20, min_periods=20).mean().values
    mad_1w = pd.Series(tp_1w).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    cci_20_1w = (tp_1w - sma_tp_1w) / (0.015 * mad_1w)
    cci_20_1w = np.where(mad_1w == 0, 0, cci_20_1w)
    
    # Align weekly CCI to daily
    cci_20_1w_aligned = align_htf_to_ltf(prices, df_1w, cci_20_1w)
    
    # Daily CCI(14) for entry/exit
    tp = (high + low + close) / 3.0
    sma_tp = pd.Series(tp).rolling(window=14, min_periods=14).mean().values
    mad = pd.Series(tp).rolling(window=14, min_periods=14).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    cci_14 = (tp - sma_tp) / (0.015 * mad)
    cci_14 = np.where(mad == 0, 0, cci_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(cci_20_1w_aligned[i]) or np.isnan(cci_14[i])):
            signals[i] = 0.0
            continue
        
        # Trend determination from weekly CCI
        uptrend = cci_20_1w_aligned[i] > 100
        downtrend = cci_20_1w_aligned[i] < -100
        
        # Entry conditions
        if uptrend and position != 1:
            # Buy dip in uptrend: CCI crosses above -100 from below
            if i > 0 and cci_14[i-1] <= -100 and cci_14[i] > -100:
                position = 1
                signals[i] = 0.25
        elif downtrend and position != -1:
            # Sell rally in downtrend: CCI crosses below 100 from above
            if i > 0 and cci_14[i-1] >= 100 and cci_14[i] < 100:
                position = -1
                signals[i] = -0.25
        # Exit conditions
        elif position == 1 and (cci_14[i] >= 100 or not uptrend):
            # Exit long on overbought or trend change
            position = 0
            signals[i] = 0.0
        elif position == -1 and (cci_14[i] <= -100 or not downtrend):
            # Exit short on oversold or trend change
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_cci_trend_reversal"
timeframe = "1d"
leverage = 1.0