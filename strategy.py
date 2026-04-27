#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike
Hypothesis: Uses daily Camarilla pivot levels (R3/S3) for breakout entries on 12h chart,
filtered by daily trend (price above/below EMA34) and volume spikes. Designed to capture
strong intraday moves with low trade frequency for robustness in both bull and bear markets.
Targets 12-37 trades per year to minimize fee drag.
"""

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
    
    # Get daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # Using previous day's OHLC to avoid look-ahead
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Shift by 1 to use previous day's data
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # First day will have invalid data (rolled from last) - will be filtered by warmup
    
    # Calculate Camarilla R3 and S3 levels
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Daily trend filter: EMA34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF levels to 12h timeframe (waits for daily close)
    R3_12h = align_htf_to_ltf(prices, df_1d, R3)
    S3_12h = align_htf_to_ltf(prices, df_1d, S3)
    ema34_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 2.0 * 24-period average (2 days)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for all indicators
    start_idx = 34  # EMA34 period
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3_12h[i]) or np.isnan(S3_12h[i]) or 
            np.isnan(ema34_12h[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3_val = R3_12h[i]
        s3_val = S3_12h[i]
        ema34_val = ema34_12h[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long conditions: price breaks above R3, above daily EMA34 (uptrend), volume spike
            if close_val > r3_val and close_val > ema34_val and vol_spike:
                signals[i] = size
                position = 1
            # Short conditions: price breaks below S3, below daily EMA34 (downtrend), volume spike
            elif close_val < s3_val and close_val < ema34_val and vol_spike:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price crosses back below R3 or below daily EMA34
            if close_val < r3_val or close_val < ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses back above S3 or above daily EMA34
            if close_val > s3_val or close_val > ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0