#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot levels (R3/S3) derived from daily candles provide strong support/resistance.
# Breakout above R3 or below S3 on 12h timeframe with volume confirmation and daily trend filter (price above/below daily EMA34).
# Works in bull markets via breakout above R3 and in bear via breakdown below S3.
# Low trade frequency expected due to multiple confirmation layers.
# Target: 12-37 trades/year on 12h timeframe.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    pivot = (high + low + close) / 3
    range_val = high - low
    r3 = close + range_val * 1.1 / 2
    s3 = close - range_val * 1.1 / 2
    return r3, s3

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels (R3, S3) on daily timeframe
    r3, s3 = calculate_camarilla(
        df_1d['high'].values, 
        df_1d['low'].values, 
        df_1d['close'].values
    )
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Camarilla levels and EMA to 12h timeframe (no look-ahead)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 12h data for entry signals
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily EMA (34) + volume EMA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r3_12h[i]) or 
            np.isnan(s3_12h[i]) or
            np.isnan(ema_34_12h[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND above daily EMA34 AND volume confirmation
            if close[i] > r3_12h[i] and close[i] > ema_34_12h[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND below daily EMA34 AND volume confirmation
            elif close[i] < s3_12h[i] and close[i] < ema_34_12h[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below R3 OR below daily EMA34
            if close[i] < r3_12h[i] or close[i] < ema_34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above S3 OR above daily EMA34
            if close[i] > s3_12h[i] or close[i] > ema_34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals