#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1wEMA50_Trend_VolumeSpike
Hypothesis: Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above R3 and close > 1w EMA50, short when price breaks below S3 and close < 1w EMA50.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Designed for 75-200 total trades over 4 years (19-50/year).
Works in both bull and bear markets by combining price structure (Camarilla) with trend (1w EMA) and volume filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from prior 1d (using prior day's range)
    # Camarilla: R4 = close + 1.1*(high-low)/2, R3 = close + 1.1*(high-low)/4,
    #            S3 = close - 1.1*(high-low)/4, S4 = close - 1.1*(high-low)/2
    # We use prior day's OHLC to avoid look-ahead
    prior_high = np.roll(high, 1)  # prior day's high
    prior_low = np.roll(low, 1)    # prior day's low
    prior_close = np.roll(close, 1) # prior day's close
    # Set first value to NaN (no prior day)
    prior_high[0] = np.nan
    prior_low[0] = np.nan
    prior_close[0] = np.nan
    
    # Calculate Camarilla R3 and S3
    camarilla_range = prior_high - prior_low
    r3 = prior_close + 1.1 * camarilla_range / 4
    s3 = prior_close - 1.1 * camarilla_range / 4
    
    # Load 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need prior day data, EMA50, volume EMA)
    start_idx = max(1, 50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Discrete position sizing
        base_size = 0.25
        
        # Long logic: price breaks above R3 + close > 1w EMA50 (trend up) + volume spike
        if close[i] > r3[i] and close[i] > ema_50_1w_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below S3 + close < 1w EMA50 (trend down) + volume spike
        elif close[i] < s3[i] and close[i] < ema_50_1w_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit conditions: price re-enters Camarilla H-L range or loss of volume confirmation
        elif position == 1 and (close[i] < (prior_high[i] + prior_low[i])/2 or not volume_spike[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > (prior_high[i] + prior_low[i])/2 or not volume_spike[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0