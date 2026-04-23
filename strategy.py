#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla Pivot R3/S3 Breakout with 1w EMA50 Trend Filter and Volume Spike
- Uses Camarilla pivot levels (R3/S3) from 4h for high-probability breakout signals
- 1w EMA50 defines long-term trend: only long when price > EMA50, short when price < EMA50
- Volume confirmation (> 1.5x 20-period average) filters weak breakouts
- Designed for 4h timeframe targeting 30-60 trades/year (120-240 over 4 years)
- Works in both bull and bear markets by following the 1w EMA50 trend filter
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
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need 1w EMA50, Camarilla
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla pivot levels (R3, S3) on 4h data up to previous bar
        if i >= 1:
            # Use previous bar's OHLC to calculate today's pivot levels (avoid look-ahead)
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            
            pivot = (prev_high + prev_low + prev_close) / 3
            range_val = prev_high - prev_low
            
            # Camarilla levels
            r3 = pivot + (range_val * 1.1 / 4)  # R3 = pivot + 1.1*(H-L)/4
            s3 = pivot - (range_val * 1.1 / 4)  # S3 = pivot - 1.1*(H-L)/4
        else:
            # Not enough data for Camarilla yet
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND above 1w EMA50 AND volume spike
            if (close[i] > r3 and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 AND below 1w EMA50 AND volume spike
            elif (close[i] < s3 and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to pivot point OR crosses 1w EMA50
            exit_signal = False
            
            if position == 1:
                # Exit long when price < pivot OR < 1w EMA50
                if close[i] < pivot or close[i] < ema_50_1w_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price > pivot OR > 1w EMA50
                if close[i] > pivot or close[i] > ema_50_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0