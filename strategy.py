#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot breakout with 12h trend filter and volume spike.
Long when price breaks above R3 + 12h EMA50 trend up + volume > 2x average.
Short when price breaks below S3 + 12h EMA50 trend down + volume > 2x average.
Exit when price returns to pivot point (mean reversion) or trend changes.
Designed for low trade frequency (~20-40/year) to minimize fee drag in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate Camarilla levels from previous day
    # Use daily high/low/close from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values  # Previous day high
    prev_low = df_1d['low'].shift(1).values    # Previous day low
    prev_close = df_1d['close'].shift(1).values # Previous day close
    
    # Align to 4h timeframe
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Calculate Camarilla levels
    range_val = prev_high_aligned - prev_low_aligned
    # Camarilla levels: H = close + range * 1.1/2, L = close - range * 1.1/2
    # R3 = close + range * 1.1/2 * 1.1 = close + range * 1.21
    # S3 = close - range * 1.1/2 * 1.1 = close - range * 1.21
    # Actually, standard Camarilla:
    # R4 = close + range * 1.1/2
    # R3 = close + range * 1.1/4
    # R2 = close + range * 1.1/6
    # R1 = close + range * 1.1/12
    # PP = (high + low + close) / 3
    # S1 = close - range * 1.1/12
    # S2 = close - range * 1.1/6
    # S3 = close - range * 1.1/4
    # S4 = close - range * 1.1/2
    
    # Using common Camarilla calculation for intraday:
    pivot = (prev_high_aligned + prev_low_aligned + prev_close_aligned) / 3.0
    range_val = prev_high_aligned - prev_low_aligned
    
    # Key levels for breakout: R3 and S3
    r3 = pivot + range_val * 1.1 / 4
    s3 = pivot - range_val * 1.1 / 4
    
    # Calculate average volume for confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(pivot[i]) or np.isnan(avg_volume[i]) or volume[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_12h_val = ema_12h_aligned[i]
        r3_val = r3[i]
        s3_val = s3[i]
        pivot_val = pivot[i]
        volume_confirm = volume[i] > 2.0 * avg_volume[i]
        
        # Trend direction from 12h EMA
        trend_up = close[i] > ema_12h_val
        trend_down = close[i] < ema_12h_val
        
        if position == 0:
            # Long: Price breaks above R3 + uptrend + volume spike
            if (close[i] > r3_val and close[i-1] <= r3_val and 
                trend_up and volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 + downtrend + volume spike
            elif (close[i] < s3_val and close[i-1] >= s3_val and 
                  trend_down and volume_confirm):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to pivot or trend changes
                if close[i] <= pivot_val or not trend_up:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to pivot or trend changes
                if close[i] >= pivot_val or not trend_down:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3_S3_Breakout_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0