#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeFilter
Hypothesis: Camarilla R3/S3 levels on 1d timeframe with 1w EMA50 trend filter and volume confirmation (>1.5x 20-period MA). 
Long when price breaks above R3 with uptrend and volume spike. Short when price breaks below S3 with downtrend and volume spike.
Uses discrete position sizing (0.30) to minimize fee churn. Designed to capture strong breakouts in both bull and bear markets 
by following the 1w trend, which adapts to regime changes. Target: 7-25 trades/year (30-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_R3 = np.zeros(len(df_1d))
    camarilla_S3 = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        if i == 0:
            camarilla_R3[i] = np.nan
            camarilla_S3[i] = np.nan
        else:
            # Camarilla formulas: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
            camarilla_R3[i] = close_1d[i-1] + 1.1 * (high_1d[i-1] - low_1d[i-1]) / 2
            camarilla_S3[i] = close_1d[i-1] - 1.1 * (high_1d[i-1] - low_1d[i-1]) / 2
    
    # Align Camarilla levels to 1d timeframe (already aligned by get_htf_data)
    camarilla_R3_1d = camarilla_R3
    camarilla_S3_1d = camarilla_S3
    
    # Align to 1d timeframe (no additional delay needed as levels are based on previous day)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3_1d)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3_1d)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    uptrend_1w = close > ema_50_1w_aligned
    downtrend_1w = close < ema_50_1w_aligned
    
    # Volume confirmation: volume > 1.5x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 1w EMA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        if position == 0:
            # Long: price breaks above R3 with 1w uptrend and volume spike
            if (close[i] > camarilla_R3_aligned[i] and 
                uptrend_1w[i] and volume_spike[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S3 with 1w downtrend and volume spike
            elif (close[i] < camarilla_S3_aligned[i] and 
                  downtrend_1w[i] and volume_spike[i]):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Exit: 1w trend changes to downtrend OR price closes below R3 (failed breakout)
            if (not uptrend_1w[i] or close[i] < camarilla_R3_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Exit: 1w trend changes to uptrend OR price closes above S3 (failed breakout)
            if (not downtrend_1w[i] or close[i] > camarilla_S3_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0