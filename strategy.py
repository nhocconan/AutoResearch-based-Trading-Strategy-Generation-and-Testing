#!/usr/bin/env python3
# 12H_Camarilla_R3_S3_Breakout_1wTrend_VolumeConfirm
# Hypothesis: Camarilla pivot breakouts (R3/S3) on 12h chart with weekly trend filter (price > EMA50 weekly)
# and volume confirmation (volume > 1.5x 20-period average) capture institutional-level moves.
# Weekly trend filter ensures alignment with higher timeframe direction, reducing false signals.
# Designed for low trade frequency (~15-30/year) with discrete sizing (0.25) to minimize fee drag.
# Works in bull markets (breakouts above R3 in uptrend) and bear markets (breakdowns below S3 in downtrend).

name = "12H_Camarilla_R3_S3_Breakout_1wTrend_VolumeConfirm"
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
    
    # Calculate typical price for Camarilla levels
    typical_price = (high + low + close) / 3.0
    
    # Get weekly data for pivot calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from weekly OHLC
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    camarilla_width = (high_1w - low_1w) * 1.1 / 2.0
    r3_level = close_1w + camarilla_width
    s3_level = close_1w - camarilla_width
    
    # Align weekly Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3_level)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3_level)
    
    # Weekly trend filter: EMA 50 on weekly closes
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_threshold[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        is_uptrend = close_1w[i] > ema_50_1w_aligned[i]  # Use weekly close for trend
        is_downtrend = close_1w[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long entry: Price breaks above R3 + volume confirmation + weekly uptrend
            if close[i] > r3_aligned[i] and volume[i] > vol_threshold[i] and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below S3 + volume confirmation + weekly downtrend
            elif close[i] < s3_aligned[i] and volume[i] > vol_threshold[i] and is_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below S3 or weekly trend turns down
            if close[i] < s3_aligned[i] or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above R3 or weekly trend turns up
            if close[i] > r3_aligned[i] or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals