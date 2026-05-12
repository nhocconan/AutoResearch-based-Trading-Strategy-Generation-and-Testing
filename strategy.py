#!/usr/bin/env python3
# 12H_CAMARILLA_R3_S3_BREAKOUT_1D_TREND_FILTER
# Hypothesis: Camarilla pivot levels (R3/S3) from 1d candles act as strong support/resistance.
# In 1d uptrend (EMA34), go long when price breaks above R3; in downtrend, go short when breaks below S3.
# Uses volume confirmation to avoid false breakouts. Works in both bull and bear markets by following 1d trend.
# Target: 15-25 trades/year on 12h timeframe.

name = "12H_CAMARILLA_R3_S3_BREAKOUT_1D_TREND_FILTER"
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
    
    # Daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar: R3, S3
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    rng = df_1d['high'] - df_1d['low']
    r3 = df_1d['close'] + 1.1 * rng / 2
    s3 = df_1d['close'] - 1.1 * rng / 2
    
    # EMA34 for trend filter
    ema34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume average (20-period) for confirmation
    vol_ma = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    
    # Align to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_ma_aligned[i]) or
            np.isnan(volume[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: 1d uptrend + price breaks above R3 + volume confirmation
            if (close[i] > ema34_aligned[i] and 
                high[i] > r3_aligned[i] and 
                volume[i] > vol_ma_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: 1d downtrend + price breaks below S3 + volume confirmation
            elif (close[i] < ema34_aligned[i] and 
                  low[i] < s3_aligned[i] and 
                  volume[i] > vol_ma_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or price re-enters R3-S3 range
            if (close[i] <= ema34_aligned[i] or 
                close[i] < r3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or price re-enters R3-S3 range
            if (close[i] >= ema34_aligned[i] or 
                close[i] > s3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals