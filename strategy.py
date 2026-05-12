#!/usr/bin/env python3
# 1D_CAMARILLA_R3_S3_BREAKOUT_WEEKLYTREND_VOLUME_CONFIRMATION
# Hypothesis: Camarilla pivot levels (R3/S3) from daily price action combined with weekly trend filter (EMA34) and volume confirmation.
# In uptrend (price > weekly EMA34), go long when price breaks above R3 with volume > 1.5x average.
# In downtrend (price < weekly EMA34), go short when price breaks below S3 with volume > 1.5x average.
# Works in both bull and bear markets: trend filter ensures we trade with the higher timeframe momentum.
# Weekly trend reduces whipsaws, volume confirmation ensures breakout legitimacy.
# Target: 15-25 trades/year on 1d timeframe.

name = "1D_CAMARILLA_R3_S3_BREAKOUT_WEEKLYTREND_VOLUME_CONFIRMATION"
timeframe = "1d"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Weekly EMA for trend filter (34-period)
    ema34 = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1w, ema34)
    
    # Daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    r3 = prev_close + 1.1 * (prev_high - prev_low)
    s3 = prev_close - 1.1 * (prev_high - prev_low)
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure volume MA is ready
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema34_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 in uptrend with volume confirmation
            if (close[i] > r3_aligned[i] and close[i] > ema34_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 in downtrend with volume confirmation
            elif (close[i] < s3_aligned[i] and close[i] < ema34_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to or goes below R3
            if close[i] <= r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to or goes above S3
            if close[i] >= s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals