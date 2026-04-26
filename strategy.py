#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: On 6h timeframe, trade long when price breaks above Camarilla R3 level with volume spike and 1d uptrend, 
short when breaks below S3 level with volume spike and 1d downtrend. Camarilla R3/S3 represent stronger support/resistance 
than R1/S1, reducing false breakouts. Volume spike confirms institutional participation. 1d EMA50 trend filter ensures 
trading with higher timeframe momentum. Designed for 50-150 total trades over 4 years (12-37/year) with discrete sizing 
(0.25) to minimize fee drag. Works in bull/bear markets via 1d trend filter and volume confirmation.
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
    
    # Get 1d data for EMA50 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from prior 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range for prior 1d bar
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = close_1d[0]  # first bar
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - prev_close_1d), np.abs(low_1d - prev_close_1d)))
    atr_1d = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values  # Wilder's ATR
    
    # Camarilla levels: based on prior day's range
    hl_range_1d = high_1d - low_1d
    r3_1d = close_1d + 1.5 * hl_range_1d  # R3 level
    s3_1d = close_1d - 1.5 * hl_range_1d  # S3 level
    
    # Align HTF indicators to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of EMA50 (50), volume MA (20)
    start_idx = max(50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_val = ema_50_1d_aligned[i]
        r3_val = r3_1d_aligned[i]
        s3_val = s3_1d_aligned[i]
        close_val = close[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: price breaks above R3, volume spike, above 1d EMA50
            long_signal = (close_val > r3_val) and vol_spike_val and (close_val > ema_50_val)
            
            # Short: price breaks below S3, volume spike, below 1d EMA50
            short_signal = (close_val < s3_val) and vol_spike_val and (close_val < ema_50_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below S3 (contrarian exit) OR volume spike in opposite direction
            if (close_val < s3_val) or (vol_spike_val and close_val < ema_50_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above R3 (contrarian exit) OR volume spike in opposite direction
            if (close_val > r3_val) or (vol_spike_val and close_val > ema_50_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0