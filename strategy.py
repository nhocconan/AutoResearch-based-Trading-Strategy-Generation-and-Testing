#!/usr/bin/env python3
# 12H_CAMARILLA_R3_S3_BREAKOUT_1D_TREND_FILTER
# Hypothesis: Camarilla pivot levels (R3/S3) from 1d act as strong support/resistance.
# In 1d uptrend (EMA34), go long when price breaks above R3; in downtrend, go short when price breaks below S3.
# Uses volume confirmation to avoid false breakouts. Trend filter ensures we trade with the higher timeframe trend.
# Designed for 12h timeframe to target 15-30 trades/year, avoiding overtrading.

name = "12H_CAMARILLA_R3_S3_BREAKOUT_1D_TREND_FILTER"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    # R3 = Close + 1.1 * (High - Low)
    # S3 = Close - 1.1 * (High - Low)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    rango = high_1d - low_1d
    r3 = close_1d + 1.1 * rango
    s3 = close_1d - 1.1 * rango
    
    # EMA34 for trend filter
    ema34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_ok = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: 1d uptrend + price breaks above R3 + volume confirmation
            if (close[i] > ema34_aligned[i] and 
                close[i] > r3_aligned[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: 1d downtrend + price breaks below S3 + volume confirmation
            elif (close[i] < ema34_aligned[i] and 
                  close[i] < s3_aligned[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Trend reversal or price falls back below R3
            if (close[i] <= ema34_aligned[i] or 
                close[i] <= r3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Trend reversal or price rises back above S3
            if (close[i] >= ema34_aligned[i] or 
                close[i] >= s3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals