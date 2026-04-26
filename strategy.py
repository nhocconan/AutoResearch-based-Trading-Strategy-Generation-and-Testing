#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeConfirm_v2
Hypothesis: Daily Camarilla R3/S3 breakout with weekly EMA50 trend filter and volume confirmation.
Only long when price breaks above R3 and close > weekly EMA50, short when price breaks below S3 and close < weekly EMA50.
Uses discrete position sizing (0.0, ±0.30) to minimize fee churn. Designed for 30-100 total trades over 4 years (7-25/year).
Works in both bull and bear markets by combining price channel breakout (Camarilla) with trend (weekly EMA) and volume filters.
BTC/ETH focus: Camarilla levels adapt to volatility, weekly trend filter avoids counter-trend trades in bear markets.
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
    
    # Calculate daily Camarilla levels (using previous day's range)
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    # Using previous bar's high/low/close to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # seed first value
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    daily_range = prev_high - prev_low
    camarilla_r3 = prev_close + 1.1 * daily_range / 2
    camarilla_s3 = prev_close - 1.1 * daily_range / 2
    
    # Load weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 1.5 * 20-period EMA volume
    avg_volume = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for weekly EMA, 1 for Camarilla prev values, 20 for volume)
    start_idx = max(50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        # Discrete position sizing
        base_size = 0.30
        
        # Long logic: price breaks above R3 + close > weekly EMA50 (trend up) + volume spike
        if close[i] > camarilla_r3[i] and close[i] > ema_50_1w_aligned[i] and volume_spike[i]:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        # Short logic: price breaks below S3 + close < weekly EMA50 (trend down) + volume spike
        elif close[i] < camarilla_s3[i] and close[i] < ema_50_1w_aligned[i] and volume_spike[i]:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        # Exit conditions: price returns to Camarilla H-L range or loss of volume confirmation
        elif position == 1 and (close[i] < camarilla_r3[i] or not volume_spike[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close[i] > camarilla_s3[i] or not volume_spike[i]):
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

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeConfirm_v2"
timeframe = "1d"
leverage = 1.0