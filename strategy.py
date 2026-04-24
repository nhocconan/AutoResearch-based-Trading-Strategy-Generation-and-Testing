#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot breakout with 1d EMA50 trend filter and volume spike confirmation.
- Uses 6h timeframe (primary) and 1d HTF for EMA50 trend alignment
- Camarilla levels calculated from previous 1d OHLC: R3, S3, R4, S4
- Long when price closes above R3 AND 1d EMA50 uptrend AND volume > 2.5 * volume MA(20)
- Short when price closes below S3 AND 1d EMA50 downtrend AND volume > 2.5 * volume MA(20)
- Exit when price reverts to the 1d VWAP (mean reversion to fair value)
- Discrete signal size: 0.25 to balance return and fee drag
- Target: 50-150 total trades over 4 years (12-37/year) as per 6h recommendation
- Works in bull/bear: trend filter prevents counter-trend trades, Camarilla levels adapt to volatility
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
    
    # Calculate 1d EMA50 for trend filter (using previous completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough data for EMA50
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d VWAP for exit (mean reversion target)
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d_values = vwap_1d.values
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d_values)
    
    # Calculate Camarilla levels from previous 1d OHLC
    # R4 = close + 1.5*(high-low), R3 = close + 1.0*(high-low)
    # S3 = close - 1.0*(high-low), S4 = close - 1.5*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3 = close_1d + 1.0 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.0 * (high_1d - low_1d)
    
    # Align Camarilla levels to 6h timeframe (previous 1d levels available at open)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current volume > 2.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.5 * volume_ma)
    
    # Trend filter: price above/below 1d EMA50
    uptrend = close > ema_50_1d_aligned
    downtrend = close < ema_50_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 1d EMA50, volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vwap_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price closes above R3 AND uptrend AND volume confirmation
            if close[i] > camarilla_r3_aligned[i] and uptrend[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price closes below S3 AND downtrend AND volume confirmation
            elif close[i] < camarilla_s3_aligned[i] and downtrend[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverts to 1d VWAP
            if close[i] < vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverts to 1d VWAP
            if close[i] > vwap_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3S3_1dEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0