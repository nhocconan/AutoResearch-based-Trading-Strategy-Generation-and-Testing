#!/usr/bin/env python3
# 12h_KAMA_Direction_With_Volume
# Hypothesis: KAMA(14,2,30) tracks price with less lag than SMA, adapting to volatility.
# Use KAMA direction for trend bias, enter on pullbacks in trend direction with volume confirmation.
# Works in bull via buying dips in uptrend, in bear via selling rallies in downtrend.
# Target: 15-25 trades/year on 12h timeframe to minimize fee drag.

name = "12h_KAMA_Direction_With_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for KAMA calculation (more stable than intraday)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for trend filter (more robust)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA on daily close
    close_1d = df_1d['close'].values
    direction = np.abs(np.diff(close_1d, 10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # not quite right, let's do properly
    
    # Proper KAMA calculation
    change = np.abs(np.diff(close_1d, 10))  # absolute net change over 10 periods
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # this needs fixing
    
    # Let's do ER (efficiency ratio) properly
    er = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        direction_val = np.abs(close_1d[i] - close_1d[i-10])
        volatility_val = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
        if volatility_val > 0:
            er[i] = direction_val / volatility_val
        else:
            er[i] = 0
    er[0:10] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # for EMA(2)
    slow_sc = 2 / (30 + 1)  # for EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # KAMA direction (slope)
    kama_direction = np.diff(kama, prepend=kama[0])
    
    # Weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_slope_20_1w = np.diff(ema_20_1w, prepend=ema_20_1w[0])
    
    # ATR for volatility and stop
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr3 = np.absolute(np.roll(close, 1) - np.roll(close, 1))  # fixed
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation (1.5x 30-period average)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    # Align indicators to 12h timeframe
    kama_dir_aligned = align_htf_to_ltf(prices, df_1d, kama_direction)
    ema_slope_aligned = align_htf_to_ltf(prices, df_1w, ema_slope_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama_dir_aligned[i]) or
            np.isnan(ema_slope_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filters
        bullish = kama_dir_aligned[i] > 0 and ema_slope_aligned[i] > 0
        bearish = kama_dir_aligned[i] < 0 and ema_slope_aligned[i] < 0
        
        # Volume confirmation
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: KAMA up, weekly trend up, volume surge
            if bullish and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, weekly trend down, volume surge
            elif bearish and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit: KAMA turns down OR weekly trend turns down
                if kama_dir_aligned[i] <= 0 or ema_slope_aligned[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit: KAMA turns up OR weekly trend turns up
                if kama_dir_aligned[i] >= 0 or ema_slope_aligned[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals