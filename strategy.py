#!/usr/bin/env python3
# 12H_CAMARILLA_R3_S3_REVERSION_1D_TREND_FILTER
# Hypothesis: Camarilla R3/S3 levels from daily pivots act as strong support/resistance.
# In uptrend (price > daily EMA34), buy at S3 reversion; in downtrend (price < daily EMA34), sell at R3 reversion.
# Works in both bull and bear markets: trend filter prevents counter-trend trades, reversion captures pullbacks.
# Target: 20-40 trades/year on 12h timeframe.

name = "12H_CAMARILLA_R3_S3_REVERSION_1D_TREND_FILTER"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    ph = df_1d['high'].shift(1).values
    pl = df_1d['low'].shift(1).values
    pc = df_1d['close'].shift(1).values
    
    # Camarilla levels: R3/S3
    r3 = pc + (ph - pl) * 1.1 / 4
    s3 = pc - (ph - pl) * 1.1 / 4
    
    # Daily EMA for trend filter (34-period)
    ema34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price at S3 in uptrend (reversion to mean)
            if (close[i] <= s3_aligned[i] and close[i] > ema34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price at R3 in downtrend (reversion to mean)
            elif (close[i] >= r3_aligned[i] and close[i] < ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches midpoint or stops trending
            if close[i] >= (r3_aligned[i] + s3_aligned[i]) / 2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches midpoint or stops trending
            if close[i] <= (r3_aligned[i] + s3_aligned[i]) / 2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals