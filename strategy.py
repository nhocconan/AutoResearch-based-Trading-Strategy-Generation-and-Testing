#!/usr/bin/env python3
# 12h_TRIX_1dTrend_Volume
# Hypothesis: Use TRIX (12-period) on 12h for momentum, filtered by 1d EMA34 trend and volume spike.
# TRIX > 0 indicates bullish momentum, TRIX < 0 bearish. Enter long when TRIX crosses above 0
# with price above 1d EMA34 and volume above average; enter short on cross below 0 with price
# below 1d EMA34 and volume confirmation. Exit on TRIX cross back through zero or trend failure.
# Designed for low frequency (15-35 trades/year) to avoid fee drag. Works in bull (catch momentum)
# and bear (catch reversals) with trend filter and volume confirmation.

name = "12h_TRIX_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(series, period):
    """Calculate Exponential Moving Average."""
    return pd.Series(series).ewm(span=period, adjust=False).mean().values

def trix(close, period=12):
    """
    Calculate TRIX indicator.
    TRIX = EMA(EMA(EMA(close, period), period), period)
    Returns percentage change: (EMA3 - EMA3_prev) / EMA3_prev * 100
    """
    ema1 = ema(close, period)
    ema2 = ema(ema1, period)
    ema3 = ema(ema2, period)
    
    # Calculate percentage change
    trix_raw = np.zeros_like(ema3)
    trix_raw[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    trix_raw[0] = 0
    
    return trix_raw

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate TRIX on 12h data
    trix_val = trix(close, 12)
    
    # Daily EMA34 for trend filter
    ema_34_1d = ema(close_1d, 34)
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align daily data to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable (34 EMA + buffer)
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(trix_val[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA34
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volume filter
        vol_ok = volume[i] > vol_ma_20[i]
        
        # TRIX signals: momentum direction
        trix_pos = trix_val[i] > 0
        trix_neg = trix_val[i] < 0
        
        # TRIX cross zero (momentum shift)
        trix_cross_up = trix_val[i] > 0 and trix_val[i-1] <= 0
        trix_cross_down = trix_val[i] < 0 and trix_val[i-1] >= 0
        
        if position == 0:
            # LONG: TRIX crosses above zero, price above daily EMA34, volume confirmation
            if trix_cross_up and trend_up and vol_ok:
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero, price below daily EMA34, volume confirmation
            elif trix_cross_down and trend_down and vol_ok:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero (momentum fails) or trend fails
            if trix_cross_down or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero (momentum fails) or trend fails
            if trix_cross_up or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals