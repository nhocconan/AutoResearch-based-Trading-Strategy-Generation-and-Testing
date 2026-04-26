#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike_v1
Hypothesis: Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation.
Only long on break above R3 when 12h EMA50 is rising and volume > 1.5x 20-bar EMA volume.
Only short on break below S3 when 12h EMA50 is falling and volume > 1.5x 20-bar EMA volume.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Designed for 75-200 total trades over 4 years (19-50/year).
Works in both bull and bear markets by combining price channel breakouts with trend and volume filters.
"""

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
    
    # Calculate Camarilla levels from previous day
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.0*(high-low), etc.
    # We need previous day's OHLC, so we'll calculate daily then align
    # For simplicity in 4h, we'll use rolling window of 6 bars (approx 1 day)
    lookback = 6  # 6 * 4h = 24h approx
    if n < lookback:
        return np.zeros(n)
    
    # Calculate rolling max/min/close for Camarilla
    rolling_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    rolling_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    rolling_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).last().values
    
    # Camarilla R3 and S3
    R3 = rolling_close + 1.0 * (rolling_high - rolling_low)
    S3 = rolling_close - 1.0 * (rolling_high - rolling_low)
    
    # Load 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(lookback, 50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or np.isnan(ema_50_12h_aligned[i]) or 
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
        
        # Long logic: break above R3 + 12h EMA50 rising + volume spike
        if close[i] > R3[i] and ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: break below S3 + 12h EMA50 falling + volume spike
        elif close[i] < S3[i] and ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit conditions: reverse breakout or loss of volume confirmation
        elif position == 1 and (close[i] < S3[i] or not volume_spike[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > R3[i] or not volume_spike[i]):
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

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0