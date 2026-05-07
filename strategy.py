#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12hTrend_Volume
Hypothesis: Camarilla pivot R3/S3 breakout with 12-hour trend filter and volume confirmation.
In bull markets (price > 12h EMA50), long on R3 breakout with volume.
In bear markets (price < 12h EMA50), short on S3 breakout with volume.
Uses 12-hour EMA50 for trend filter and volume confirmation to avoid false breakouts.
Target: 25-40 trades per year (~100-160 over 4 years) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3_S3_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12-hour data ONCE for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12-hour EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla pivot levels using previous day's data
    # We'll use the previous day's high, low, close for 4h bars
    # For simplicity, we'll approximate using rolling window of 6 bars (1.5 days) for daily OHLC
    # In practice, we'd use actual daily data, but this approximation works for the logic
    roll_high = pd.Series(high).rolling(window=6, min_periods=6).max().values
    roll_low = pd.Series(low).rolling(window=6, min_periods=6).min().values
    roll_close = pd.Series(close).rolling(window=6, min_periods=6).last().values
    
    # Calculate pivot and Camarilla levels
    pivot = (roll_high + roll_low + roll_close) / 3
    range_val = roll_high - roll_low
    r3 = pivot + (range_val * 1.1 / 4)
    s3 = pivot - (range_val * 1.1 / 4)
    
    # Volume ratio: current volume / 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need 50 periods for EMA50 and sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market regime from 12-hour EMA50
        uptrend_regime = close[i] > ema_50_12h_aligned[i]
        downtrend_regime = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation: volume > 1.5x average
        volume_confirm = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: close breaks above R3 in uptrend regime + volume
            long_entry = (close[i] > r3[i]) and uptrend_regime and volume_confirm
            # Short: close breaks below S3 in downtrend regime + volume
            short_entry = (close[i] < s3[i]) and downtrend_regime and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: close crosses below pivot or regime changes to downtrend
            if (close[i] < pivot[i]) or (not uptrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: close crosses above pivot or regime changes to uptrend
            if (close[i] > pivot[i]) or (not downtrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals