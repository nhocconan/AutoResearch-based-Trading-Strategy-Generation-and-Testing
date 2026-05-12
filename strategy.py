#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeS
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d trend filter and volume confirmation.
# Long when price breaks above R3 with 1d uptrend and volume spike; short when price breaks below S3 with 1d downtrend and volume spike.
# Exit on opposite Camarilla level touch (S3 for long, R3 for short) or trend reversal.
# Uses volume spike (1.5x 20-period average) to filter breakouts, reducing false signals.
# Designed for low frequency (12-37 trades/year) to avoid fee drag.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeS"
timeframe = "12h"
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
    
    # Calculate 12h Camarilla levels from prior bar
    # Camarilla: R3 = close + (high-low)*1.1/2, S3 = close - (high-low)*1.1/2
    # Using prior bar to avoid look-ahead
    camarilla_r3 = np.zeros(n)
    camarilla_s3 = np.zeros(n)
    for i in range(1, n):
        rang = high[i-1] - low[i-1]
        camarilla_r3[i] = close[i-1] + rang * 1.1 / 2
        camarilla_s3[i] = close[i-1] - rang * 1.1 / 2
    
    # Volume spike: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Load 1d trend data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 1d EMA20 for trend filter
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(ema20_1d_aligned[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: 1d close above/below EMA20
        trend_up = close > ema20_1d_aligned[i]
        trend_down = close < ema20_1d_aligned[i]
        
        if position == 0:
            # LONG: price breaks above R3 with uptrend and volume spike
            if high[i] > camarilla_r3[i] and trend_up and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S3 with downtrend and volume spike
            elif low[i] < camarilla_s3[i] and trend_down and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: price touches S3 or trend turns down
            if low[i] <= camarilla_s3[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price touches R3 or trend turns up
            if high[i] >= camarilla_r3[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals