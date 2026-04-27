#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_Volume_Spike
Hypothesis: Use weekly EMA34 trend filter with 12h timeframe, entering on breaks of weekly R3/S3 levels with volume confirmation. Weekly timeframe reduces noise and false signals, while volume spike confirms institutional interest. Designed to work in both bull (breakouts continue) and bear (false breakdowns reversed quickly) markets by requiring alignment with weekly trend.
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
    
    # Get weekly data for trend filter and Camarilla pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate weekly Camarilla levels (R3/S3 - stronger breakout levels)
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    range_ = df_1w['high'] - df_1w['low']
    
    # Camarilla R3 and S3 (strongest breakout levels)
    r3 = typical_price + (range_ * 1.1 / 4)
    s3 = typical_price - (range_ * 1.1 / 4)
    
    # Align levels to 12h timeframe (use previous week's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3.values)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA and volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            signals[i] = 0.0
            continue
        
        ema_trend = ema34_1w_aligned[i]
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: price closes above R3 + volume spike + uptrend (price > EMA34)
            if close[i] > r3_level and vol_spike_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: price closes below S3 + volume spike + downtrend (price < EMA34)
            elif close[i] < s3_level and vol_spike_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below S3 or trend turns down
            if close[i] < s3_level or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price closes above R3 or trend turns up
            if close[i] > r3_level or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume_Spike"
timeframe = "12h"
leverage = 1.0