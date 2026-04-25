#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_HTFConfluence
Hypothesis: Trade 6h Camarilla R3/S3 breakouts with 1d EMA50 trend filter and 1d volume confirmation. 
R3/S3 levels act as strong support/resistance - breaks indicate momentum continuation. 
1d EMA50 ensures alignment with daily trend, reducing counter-trend whipsaws. 
Volume confirmation filters low-conviction breakouts. 
Discrete sizing 0.25 balances profit and fee drag. Target: 12-25 trades/year (~50-100 over 4 years) to stay within fee drag limits for 6h timeframe.
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
    
    # Get 1d data for HTF filters
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d volume MA20 for confirmation
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Calculate Camarilla levels from previous 1d bar's OHLC
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    prev_close_1d = df_1d['close'].shift(1).values
    
    camarilla_range = prev_high_1d - prev_low_1d
    r3 = prev_close_1d + 1.1 * camarilla_range / 4   # R3 level
    s3 = prev_close_1d - 1.1 * camarilla_range / 4   # S3 level
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1d EMA50 (50), volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_ma20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND 1d trend bullish (close > EMA50) AND 1d volume > 1.2x MA20
            long_setup = (close[i] > r3_aligned[i]) and \
                         (close[i] > ema_50_1d_aligned[i]) and \
                         (volume_1d[-1] > 1.2 * vol_ma20_1d_aligned[i] if len(volume_1d) > 0 else False)
            # Short: price breaks below S3 AND 1d trend bearish (close < EMA50) AND 1d volume > 1.2x MA20
            short_setup = (close[i] < s3_aligned[i]) and \
                          (close[i] < ema_50_1d_aligned[i]) and \
                          (volume_1d[-1] > 1.2 * vol_ma20_1d_aligned[i] if len(volume_1d) > 0 else False)
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price re-enters R3/S3 range OR 1d trend turns bearish
            if (close[i] < r3_aligned[i] and close[i] > s3_aligned[i]) or \
               (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters R3/S3 range OR 1d trend turns bullish
            if (close[i] < r3_aligned[i] and close[i] > s3_aligned[i]) or \
               (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dTrend_HTFConfluence"
timeframe = "6h"
leverage = 1.0