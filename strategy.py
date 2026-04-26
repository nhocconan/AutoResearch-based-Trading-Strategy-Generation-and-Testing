#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike
Hypothesis: Camarilla R3/S3 levels from 12h timeframe act as strong support/resistance. 
Breakout above R3 with volume spike and 12h EMA50 uptrend = long. Breakdown below S3 with volume spike and 12h EMA50 downtrend = short.
Uses discrete position sizing (0.30) to minimize fee drag. Target: 20-50 trades/year per symbol.
Works in bull/bear via 12h trend filter - only long in uptrend, short in downtrend.
Volume spike filter ensures conviction. ATR-based stoploss manages risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla levels and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from 12h OHLC (use previous completed 12h bar)
    # We need to shift by 1 to avoid look-ahead - use previous bar's OHLC
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_shift = df_12h['close'].shift(1).values  # Previous bar close
    
    # Calculate Camarilla levels for each 12h bar
    rng = high_12h - low_12h
    camarilla_r3 = close_12h_shift + 1.1 * rng / 2  # R3 = Close + 1.1*(High-Low)/2
    camarilla_s3 = close_12h_shift - 1.1 * rng / 2  # S3 = Close - 1.1*(High-Low)/2
    
    # Align Camarilla levels to 4h timeframe (with proper delay for completed bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # Volume spike detector (20-bar volume MA on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        # Trend filter from 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: Break above R3 with volume spike and uptrend
            if close[i] > camarilla_r3_aligned[i] and volume_spike[i] and uptrend:
                signals[i] = 0.30
                position = 1
            # Short: Break below S3 with volume spike and downtrend
            elif close[i] < camarilla_s3_aligned[i] and volume_spike[i] and downtrend:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Exit: Close below R3 (failed breakout) OR trend change
            if close[i] < camarilla_r3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Exit: Close above S3 (failed breakdown) OR trend change
            if close[i] > camarilla_s3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0