#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # Daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily data for trend filter (EMA50)
    df_1d_ema = get_htf_data(prices, '1d')
    if len(df_1d_ema) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels using previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula: Range = High - Low
    # R3 = Close + Range * 1.1/2
    # S3 = Close - Range * 1.1/2
    range_1d = high_1d - low_1d
    r3_1d = close_1d + (range_1d * 1.1 / 2)
    s3_1d = close_1d - (range_1d * 1.1 / 2)
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d_ema['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 4h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d_ema, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price above R3, above daily EMA50, volume filter
            long_cond = (close[i] > r3_1d_aligned[i]) and (close[i] > ema_50_1d_aligned[i]) and volume_filter[i]
            # Short conditions: price below S3, below daily EMA50, volume filter
            short_cond = (close[i] < s3_1d_aligned[i]) and (close[i] < ema_50_1d_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below S3 (reversal level)
            if close[i] < s3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above R3 (reversal level)
            if close[i] > r3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 levels act as strong reversal points in both bull and bear markets.
# Using daily timeframe for pivot calculation and trend filter (EMA50) ensures alignment with institutional levels.
# Volume confirmation filters out low-conviction breakouts. Exit at opposite S3/R3 levels captures mean reversion.
# Designed for 4h timeframe to balance trade frequency (target: 20-50 trades/year) and capture significant moves.