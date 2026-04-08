#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6s Camarilla pivot with 1d trend filter and volume confirmation.
# Enter long when price breaks above R3 with 1d EMA(50) rising and volume > 1.5x avg.
# Enter short when price breaks below S3 with 1d EMA(50) falling and volume > 1.5x avg.
# Exit on opposite Camarilla breakout or when price crosses 1d EMA(50).
# Target: 50-150 total trades over 4 years (12-37/year) with controlled risk.

name = "6h_camarilla1d_ema50_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Daily Camarilla pivot levels (using previous day's OHLC)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla levels
    r3 = prev_close + 1.1 * prev_range * 1.1 / 12
    s3 = prev_close - 1.1 * prev_range * 1.1 / 12
    r4 = prev_close + 1.1 * prev_range * 1.5 / 12
    s4 = prev_close - 1.1 * prev_range * 1.5 / 12
    
    # Align daily levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price breaks below S3 OR crosses below EMA50
            if close[i] < s3_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above R3 OR crosses above EMA50
            if close[i] > r3_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Camarilla breakout + EMA50 trend + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > r3_aligned[i] and close[i] > ema_50_aligned[i]:
                    # Breakout above R3 in uptrend: long
                    signals[i] = 0.25
                    position = 1
                elif close[i] < s3_aligned[i] and close[i] < ema_50_aligned[i]:
                    # Breakdown below S3 in downtrend: short
                    signals[i] = -0.25
                    position = -1
    
    return signals