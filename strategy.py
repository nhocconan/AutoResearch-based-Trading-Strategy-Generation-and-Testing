#!/usr/bin/env python3
# 1d_Weekly_Camarilla_Pivot_Squeeze_With_Volume_Filter
# Hypothesis: Trade weekly Camarilla pivot levels (R3/S3) on daily chart with volume confirmation.
# Uses weekly trend filter (EMA34) to align with higher timeframe direction.
# Designed for low-frequency, high-conviction trades in both bull and bear markets.
# Target: 10-25 trades/year (~40-100 total over 4 years) to minimize fee drag.

name = "1d_Weekly_Camarilla_Pivot_Squeeze_With_Volume_Filter"
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
    
    # Weekly OHLC for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels: R3, S3
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    camarilla_r3 = close_1w + 1.1 * (high_1w - low_1w)
    camarilla_s3 = close_1w - 1.1 * (high_1w - low_1w)
    
    # Align weekly levels to daily
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Weekly trend filter: EMA34 on weekly close
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume confirmation: current volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches or breaks above S3, weekly uptrend, volume confirmation
            if (close[i] >= camarilla_s3_aligned[i] and 
                close[i] > ema34_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price touches or breaks below R3, weekly downtrend, volume confirmation
            elif (close[i] <= camarilla_r3_aligned[i] and 
                  close[i] < ema34_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below EMA34 or reaches R3 (take profit)
            if (close[i] < ema34_1w_aligned[i] or 
                close[i] >= camarilla_r3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above EMA34 or reaches S3 (take profit)
            if (close[i] > ema34_1w_aligned[i] or 
                close[i] <= camarilla_s3_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals