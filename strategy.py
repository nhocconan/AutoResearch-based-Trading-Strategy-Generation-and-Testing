#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_DailyTrend_VolumeSpike
Hypothesis: 12h Camarilla R3/S3 breakout with daily trend filter (price > EMA50 daily) and volume confirmation.
Works in bull/bear by using daily trend filter to avoid counter-trend trades. Target: 15-30 trades/year (60-120 total).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily trend filter (EMA50)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    close_daily = df_daily['close'].values
    ema50_daily = pd.Series(close_daily).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    # Camarilla levels from daily OHLC
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    daily_range = df_daily['high'] - df_daily['low']
    camarilla_R3 = df_daily['close'] + daily_range * 1.1 / 4
    camarilla_S3 = df_daily['close'] - daily_range * 1.1 / 4
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_R3.values)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_daily, camarilla_S3.values)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for calculations
    start_idx = max(50, 20) + 1
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_daily_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or 
            np.isnan(camarilla_S3_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price > R3 + daily uptrend + volume spike
            if (close[i] > camarilla_R3_aligned[i] and close[i] > ema50_daily_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price < S3 + daily downtrend + volume spike
            elif (close[i] < camarilla_S3_aligned[i] and close[i] < ema50_daily_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price < S3 or daily trend failure
            if (close[i] < camarilla_S3_aligned[i] or close[i] < ema50_daily_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price > R3 or daily trend failure
            if (close[i] > camarilla_R3_aligned[i] or close[i] > ema50_daily_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R3S3_DailyTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0