#!/usr/bin/env python3
# 12h_1d_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: 12h breakout at daily Camarilla R1/S1 levels with weekly trend filter and volume confirmation.
# Weekly trend avoids counter-trend trades in both bull/bear markets. Volume surge confirms institutional participation.
# Targets 12-37 trades/year per symbol (50-150 total over 4 years) with low frequency to minimize fee drag.
# Works in bull/bear by requiring weekly trend alignment and volume confirmation to reduce whipsaws.

name = "12h_1d_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily Camarilla levels (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r1 = np.full(len(df_1d), np.nan)
    camarilla_s1 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        diff = prev_high - prev_low
        
        camarilla_r1[i] = prev_close + diff * 1.1 / 6
        camarilla_s1[i] = prev_close - diff * 1.1 / 6
    
    # Calculate weekly EMA20 for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align daily indicators to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Align weekly EMA20 to 12h timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Align weekly close for trend comparison
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, df_1w['close'].values)
    
    # Volume average (20-period for 12h = 10 days)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough history for daily Camarilla + weekly EMA20 + vol MA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend: weekly close > EMA20
        uptrend = close_1w_aligned[i] > ema_20_1w_aligned[i]
        downtrend = close_1w_aligned[i] < ema_20_1w_aligned[i]
        
        # Volume confirmation (2x average for significance)
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Breakout above Camarilla R1 in uptrend with volume spike
            if close[i] > camarilla_r1_aligned[i] and uptrend and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below Camarilla S1 in downtrend with volume spike
            elif close[i] < camarilla_s1_aligned[i] and downtrend and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Long exit: close below Camarilla R1 or trend fails
                if close[i] < camarilla_r1_aligned[i] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: close above Camarilla S1 or trend fails
                if close[i] > camarilla_s1_aligned[i] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals