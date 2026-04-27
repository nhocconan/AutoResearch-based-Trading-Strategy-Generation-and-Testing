#!/usr/bin/env python3
"""
1h_CamarillaR3S3_Breakout_4hTrend_VolumeSpike
Hypothesis: Use 4h-derived Camarilla R3/S3 levels for breakout signals, filtered by 4h EMA50 trend and volume spikes (>2x 20-period average). Only take long when price > EMA50 (uptrend) and short when price < EMA50 (downtrend). Designed for 1h timeframe with 4h trend filter to reduce false breakouts and limit trades to 15-37/year. Works in bull (breakouts with trend) and bear (mean reversion at extremes with trend filter).
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
    
    # Calculate Camarilla levels from 4h timeframe
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Previous 4h bar's OHLC for Camarilla calculation
    prev_close = df_4h['close'].shift(1).values
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (wait for previous 4h bar's close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # 4h EMA50 for trend filter
    ema_50 = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    # Session filter: 08-20 UTC (only trade during active hours)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital
    
    # Warmup: need enough data for volume average and EMA
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_confirm[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        ema_50_val = ema_50_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume confirmation AND above 4h EMA50 (uptrend)
            if close[i] > camarilla_r3_val and vol_conf and close[i] > ema_50_val:
                signals[i] = size
                position = 1
            # Short: price breaks below S3 with volume confirmation AND below 4h EMA50 (downtrend)
            elif close[i] < camarilla_s3_val and vol_conf and close[i] < ema_50_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 (opposite level)
            if close[i] < camarilla_s3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above R3 (opposite level)
            if close[i] > camarilla_r3_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_CamarillaR3S3_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0