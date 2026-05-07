#!/usr/bin/env python3
"""
1h_R3S3_Breakout_4hTrend_1dVolume
Hypothesis: 1h timeframe with 4h trend filter (EMA20) and 1d volume confirmation.
Uses 4h OHLC to calculate daily-style Camarilla R3/S3 levels (based on prior 4h bar).
Breakouts occur when price penetrates R3/S3 with volume spike (1d volume > 1.5x 20-period average)
and 4h trend alignment. Designed for 15-30 trades/year to avoid fee drag in 1h.
Works in bull/bear via trend filter and volume confirmation.
"""

name = "1h_R3S3_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

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
    
    # Get 4h data for Camarilla calculation and trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h OHLC for Camarilla (using prior 4h bar)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Range and Camarilla levels from prior 4h bar
    range_4h = high_4h - low_4h
    r3_4h = close_4h + 1.1 * range_4h
    s3_4h = close_4h - 1.1 * range_4h
    
    # Align Camarilla levels to 1h
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    
    # 4h EMA20 for trend filter
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d volume average
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.divide(vol_1d, vol_ma20_1d, out=np.zeros_like(vol_1d), where=vol_ma20_1d!=0)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isclose(r3_4h_aligned[i], 0) or np.isclose(s3_4h_aligned[i], 0) or
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 4h trend
        close_4h_aligned = align_htf_to_ltf(prices, df_4h, close_4h)
        if np.isnan(close_4h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        trend_up = close_4h_aligned[i] > ema_20_4h_aligned[i]
        trend_down = close_4h_aligned[i] < ema_20_4h_aligned[i]
        
        if position == 0:
            # Long: break above R3 with volume and uptrend
            if (close[i] > r3_4h_aligned[i] and 
                vol_ratio_1d_aligned[i] > 1.5 and 
                trend_up):
                signals[i] = 0.20
                position = 1
            # Short: break below S3 with volume and downtrend
            elif (close[i] < s3_4h_aligned[i] and 
                  vol_ratio_1d_aligned[i] > 1.5 and 
                  trend_down):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: break below S3 or trend turns down
            if (close[i] < s3_4h_aligned[i] or 
                not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: break above R3 or trend turns up
            if (close[i] > r3_4h_aligned[i] or 
                not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals