#!/usr/bin/env python3
"""
1d_Weekly_R3_S3_Breakout_WeeklyTrend_VolumeSpike_v1
Hypothesis: Use weekly R3/S3 breakouts with weekly trend filter (EMA20) and daily volume confirmation.
This strategy targets fewer trades by using weekly levels, reducing frequency to ~10-25 trades/year.
Works in bull markets (buy R3 breakouts in uptrend) and bear markets (sell S3 breakdowns in downtrend).
Volume spike confirms institutional interest. Designed for BTC/ETH resilience in ranging/down markets.
"""

name = "1d_Weekly_R3_S3_Breakout_WeeklyTrend_VolumeSpike_v1"
timeframe = "1d"
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
    
    # === WEEKLY Data for R3/S3 and Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's OHLC for Camarilla calculation
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    prev_high[0] = high_1w[0]
    prev_low[0] = low_1w[0]
    prev_close[0] = close_1w[0]
    
    # Camarilla levels: R3/S3 = C ± (H-L) * 1.1/2
    rang = prev_high - prev_low
    r3 = prev_close + rang * 1.1 / 2
    s3 = prev_close - rang * 1.1 / 2
    
    # Trend filter: EMA20 on weekly close
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly indicators to daily timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # Daily volume spike: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND weekly uptrend (price > EMA20) AND volume spike
            if close[i] > r3_aligned[i] and close[i] > ema_20_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND weekly downtrend (price < EMA20) AND volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema_20_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below weekly EMA20 OR reverses below R3
            if close[i] < ema_20_aligned[i] or close[i] < r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price crosses above weekly EMA20 OR reverses above S3
            if close[i] > ema_20_aligned[i] or close[i] > s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals