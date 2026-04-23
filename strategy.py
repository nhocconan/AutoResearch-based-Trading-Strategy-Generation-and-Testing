#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation
- Camarilla pivot levels (R3, S3) act as strong intraday support/resistance levels
- Breakout above R3 or below S3 with volume confirmation captures institutional moves
- 12h EMA(50) ensures alignment with higher timeframe trend to reduce counter-trend trades
- Designed for 4h timeframe targeting 19-50 trades/year (75-200 over 4 years) to minimize fee drag
- Works in both bull and bear markets by trading with the 12h trend when price breaks key levels
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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla pivot levels from previous day (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use previous day's high, low, close for today's Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R3, S3 (most significant levels)
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    camarilla_range = (high_1d - low_1d) * 1.1 / 4
    r3 = close_1d + camarilla_range
    s3 = close_1d - camarilla_range
    
    # Align 1d Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: > 1.8x 24-period average (4h * 24 = 4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 24)  # EMA12h, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Camarilla breakout signals with trend filter
        # Long: price breaks above R3 + uptrend + volume spike
        # Short: price breaks below S3 + downtrend + volume spike
        long_signal = (close[i] > r3_aligned[i] and 
                      close[i] > ema_50_12h_aligned[i] and
                      volume[i] > 1.8 * vol_ma[i])
        
        short_signal = (close[i] < s3_aligned[i] and 
                       close[i] < ema_50_12h_aligned[i] and
                       volume[i] > 1.8 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to Camarilla midpoint or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns below midpoint (close + (high-low)*1.1/2) or trend reversal
                midpoint = close_1d[i] + (high_1d[i] - low_1d[i]) * 1.1 / 2
                midpoint_aligned = align_htf_to_ltf(prices, df_1d, np.array([midpoint]))[0] if not np.isnan(midpoint) else np.nan
                if (not np.isnan(midpoint_aligned) and close[i] < midpoint_aligned) or \
                   (close[i] < ema_50_12h_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: price returns above midpoint or trend reversal
                midpoint = close_1d[i] - (high_1d[i] - low_1d[i]) * 1.1 / 2
                midpoint_aligned = align_htf_to_ltf(prices, df_1d, np.array([midpoint]))[0] if not np.isnan(midpoint) else np.nan
                if (not np.isnan(midpoint_aligned) and close[i] > midpoint_aligned) or \
                   (close[i] > ema_50_12h_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0