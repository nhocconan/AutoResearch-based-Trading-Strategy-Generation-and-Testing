#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: 4h breakout at weekly Camarilla R1/S1 with weekly trend filter and volume confirmation.
# Uses weekly trend (close > EMA50) to avoid counter-trend trades in bear markets.
# Volume surge (2x 24-period MA) confirms institutional participation.
# Designed for 4h timeframe targeting 20-50 trades/year per symbol.
# Works in bull/bear by requiring trend alignment and volume confirmation to reduce whipsaws.

name = "4h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate weekly Camarilla levels (using previous week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    camarilla_r1 = np.full(len(df_1w), np.nan)
    camarilla_s1 = np.full(len(df_1w), np.nan)
    
    for i in range(1, len(df_1w)):
        prev_high = high_1w[i-1]
        prev_low = low_1w[i-1]
        prev_close = close_1w[i-1]
        diff = prev_high - prev_low
        
        camarilla_r1[i] = prev_close + diff * 1.1 / 6
        camarilla_s1[i] = prev_close - diff * 1.1 / 6
    
    # Align weekly indicators to 4h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # Volume average (24-period for 4h = 4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough history for weekly EMA50 + vol MA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend: weekly close > EMA50
        close_1w_aligned = align_htf_to_ltf(prices, df_1w, df_1w['close'].values)
        uptrend = close_1w_aligned[i] > ema_50_1w_aligned[i]
        downtrend = close_1w_aligned[i] < ema_50_1w_aligned[i]
        
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