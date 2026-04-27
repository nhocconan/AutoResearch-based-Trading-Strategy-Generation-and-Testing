#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_12hTrend_Volume_Strict_v1
Hypothesis: Combines Camarilla pivot levels (R3/S3) with 12h trend filter and volume spike.
This strategy targets institutional levels with trend alignment and volume confirmation to capture
strong directional moves while minimizing false breakouts. Designed for low trade frequency
(20-50 trades/year) to reduce fee drag in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate daily pivot points and Camarilla levels
    # Use 1d data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and Camarilla levels for each day
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3_1d = close_1d + (range_1d * 1.1 / 2.0)  # R3 = C + (H-L)*1.1/2
    s3_1d = close_1d - (range_1d * 1.1 / 2.0)  # S3 = C - (H-L)*1.1/2
    
    # 12h trend filter: EMA50
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    # Align all indicators to 4h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for calculations
    start_idx = 50  # EMA50 needs 50 periods
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        ema50_12h = ema50_12h_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        if position == 0:
            # Determine trend: price vs EMA50 (12h)
            uptrend = close_val > ema50_12h
            downtrend = close_val < ema50_12h
            
            if uptrend and vol_conf:
                # Long: break above R3 with volume
                if close_val > r3:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif downtrend and vol_conf:
                # Short: break below S3 with volume
                if close_val < s3:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit: price re-enters below R3 or trend reversal
            if close_val < r3:  # Re-enter below R3
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: price re-enters above S3 or trend reversal
            if close_val > s3:  # Re-enter above S3
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hTrend_Volume_Strict_v1"
timeframe = "4h"
leverage = 1.0