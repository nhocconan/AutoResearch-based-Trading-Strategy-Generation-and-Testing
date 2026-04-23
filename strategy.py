#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation
- Uses Camarilla pivot levels (R3, S3) from 1d HTF for breakout signals
- 1w EMA(50) as trend filter: long when price > EMA50, short when price < EMA50
- Volume confirmation (> 1.5x 20-period average) filters low-momentum breakouts
- Designed for 12h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by aligning with weekly trend
- Tight entry conditions to minimize fee drag and overtrading
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations: based on previous day's OHLC
    camarilla_high = np.zeros_like(close_1d)
    camarilla_low = np.zeros_like(close_1d)
    
    for i in range(1, len(close_1d)):
        # Previous day's range
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        prev_range = prev_high - prev_low
        
        # Camarilla levels
        camarilla_high[i] = prev_close + 1.1 * prev_range / 2  # R3
        camarilla_low[i] = prev_close - 1.1 * prev_range / 2   # S3
    
    # Align Camarilla levels to 12h timeframe
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    
    # Calculate 1w EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(1, 50, 20)  # Camarilla, EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine breakout conditions
        price_above_r3 = close[i] > camarilla_high_aligned[i]
        price_below_s3 = close[i] < camarilla_low_aligned[i]
        
        # Trend filter: EMA50 direction
        above_ema = close[i] > ema_50_aligned[i]
        below_ema = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Long conditions: price breaks above R3, above EMA50, volume spike
            long_signal = (price_above_r3 and 
                          above_ema and
                          volume[i] > 1.5 * vol_ma[i])
            
            # Short conditions: price breaks below S3, below EMA50, volume spike
            short_signal = (price_below_s3 and 
                           below_ema and
                           volume[i] > 1.5 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Camarilla breakout or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below S3 or trend turns bearish
                if (price_below_s3 or 
                    below_ema):
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above R3 or trend turns bullish
                if (price_above_r3 or 
                    above_ema):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1wEMA50_Trend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0