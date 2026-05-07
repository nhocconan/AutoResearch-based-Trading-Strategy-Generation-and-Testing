#!/usr/bin/env python3
# 4h_TRIX_VolumeSpike_1dTrend
# Hypothesis: Uses TRIX (12-period) on 1d timeframe to detect momentum, filtered by 1d EMA50 trend and volume spikes.
# TRIX filters out insignificant cycles and highlights significant momentum shifts.
# Works in both bull and bear markets by only trading in the direction of the 1d trend.
# Target: 20-40 trades/year to stay within optimal frequency range and minimize fee drag.

name = "4h_TRIX_VolumeSpike_1dTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for TRIX calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate TRIX (12-period) on 1d data
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) then % change
    period = 12
    ema1 = pd.Series(close_1d).ewm(span=period, adjust=False, min_periods=period).mean()
    ema2 = ema1.ewm(span=period, adjust=False, min_periods=period).mean()
    ema3 = ema2.ewm(span=period, adjust=False, min_periods=period).mean()
    trix_raw = ema3.pct_change() * 100  # Percentage change
    
    # Convert to numpy array, handling NaN from pct_change
    trix = trix_raw.fillna(0).values
    
    # Align TRIX to 4h timeframe
    trix_4h = align_htf_to_ltf(prices, df_1d, trix)
    
    # Calculate volume spike on 4h timeframe (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(trix_4h[i]) or np.isnan(ema_50_1d_4h[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX > 0 (bullish momentum) + above 1d EMA50 + volume spike
            if trix_4h[i] > 0 and close[i] > ema_50_1d_4h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: TRIX < 0 (bearish momentum) + below 1d EMA50 + volume spike
            elif trix_4h[i] < 0 and close[i] < ema_50_1d_4h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX crosses below 0 (momentum fading) or price closes below 1d EMA50
            if trix_4h[i] < 0 or close[i] < ema_50_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX crosses above 0 (momentum fading) or price closes above 1d EMA50
            if trix_4h[i] > 0 or close[i] > ema_50_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals