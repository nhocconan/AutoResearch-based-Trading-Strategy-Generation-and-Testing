#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation.
- Uses 4h timeframe (primary) and 12h HTF for EMA50 trend alignment
- Camarilla levels calculated from prior 1d OHLC: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
- Breakout logic: long when price crosses above R3 with volume confirmation, short when price crosses below S3
- Trend filter: only long when price > 12h EMA50, only short when price < 12h EMA50
- Volume confirmation: current volume > 1.5 * 20-period volume MA to avoid low-volume false signals
- Exit: reverse signal or when price reverts to prior 12h VWAP (mean reversion)
- Discrete signal size: 0.25 to balance return and risk
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe as per research
- Works in both bull/bear: trend filter avoids counter-trend trades, Camarilla breakouts capture momentum in all regimes
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate prior 1d Camarilla R3/S3 levels (using prior day's data to avoid look-ahead)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior 1d OHLC (shifted by 1 to use completed day only)
    high_1d = df_1d['high'].shift(1).values
    low_1d = df_1d['low'].shift(1).values
    close_1d = df_1d['close'].shift(1).values
    
    # Camarilla R3 and S3: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate prior 12h VWAP for mean reversion exit
    typical_price_12h = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    volume_12h = df_12h['volume']
    vwap_12h = (typical_price_12h * volume_12h).cumsum() / volume_12h.cumsum()
    vwap_12h_values = vwap_12h.values
    vwap_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h_values)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * volume_ma)
    
    # Trend filter: price above/below 12h EMA50
    uptrend = close > ema_50_12h_aligned
    downtrend = close < ema_50_12h_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 12h EMA50 and volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vwap_aligned[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above R3 AND uptrend AND volume confirmation
            if close[i] > r3_aligned[i] and close[i-1] <= r3_aligned[i-1] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below S3 AND downtrend AND volume confirmation
            elif close[i] < s3_aligned[i] and close[i-1] >= s3_aligned[i-1] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to prior 12h VWAP (mean reversion) or reverse signal
            if close[i] <= vwap_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to prior 12h VWAP (mean reversion) or reverse signal
            if close[i] >= vwap_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_12hEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0