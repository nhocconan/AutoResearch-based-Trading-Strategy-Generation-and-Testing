#!/usr/bin/env python3
"""
4H_CAMARILLA_R3_S3_BREAKOUT_1D_TREND_VOLUME_SPIKE
Hypothesis: Camarilla R3/S3 breakout on 4h with 1d trend filter (EMA34) and volume spike (>2x 20-day avg). 
Only take long when price > EMA34, short when price < EMA34. Volume spike confirms institutional participation.
Designed for fewer, high-quality trades in both bull and bear markets.
"""
name = "4H_CAMARILLA_R3_S3_BREAKOUT_1D_TREND_VOLUME_SPIKE"
timeframe = "4h"
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
    
    # Camarilla levels (based on previous day's OHLC)
    # We'll use 1d data to calculate Camarilla for each day, then align to 4h
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3 and S3
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h (no delay needed as they're based on prior day's close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 1d trend filter: EMA34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current day volume > 2x 20-day average
    vol_1d = df_1d['volume'].values
    vol_ma_20d = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = vol_1d > (2.0 * vol_ma_20d)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1st bar
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price > Camarilla R3, above EMA34 trend, and volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema34_aligned[i] and 
                vol_spike_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price < Camarilla S3, below EMA34 trend, and volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema34_aligned[i] and 
                  vol_spike_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Camarilla S3 or EMA34
            if close[i] < s3_aligned[i] or close[i] < ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Camarilla R3 or EMA34
            if close[i] > r3_aligned[i] or close[i] > ema34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals