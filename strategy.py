# 6h_Camarilla_R3S3_Breakout_1wTrend_Volume
# Hypothesis: Use weekly trend direction (from 1w EMA50) to filter Camarilla R3/S3 breakouts on 6h.
# In weekly uptrend, look for long breakouts above R3; in weekly downtrend, look for short breakdowns below S3.
# Volume confirmation ensures breakouts are genuine.
# Weekly trend filter avoids trading against the major trend, reducing losses in bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull markets via long breakouts in uptrend, in bear markets via short breakdowns in downtrend.

name = "6h_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous 1d
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # True range for width calculation
    tr = np.maximum(prev_high - prev_low, 
                    np.maximum(np.abs(prev_high - np.roll(prev_close, 1)), 
                               np.abs(prev_low - np.roll(prev_close, 1))))
    tr[0] = 0  # first value has no previous day
    
    # Calculate Camarilla levels for each day
    h4 = prev_close + 1.1/12 * tr  # Resistance 4
    l4 = prev_close - 1.1/12 * tr  # Support 4
    h3 = prev_close + 1.1/6 * tr   # Resistance 3
    l3 = prev_close - 1.1/6 * tr   # Support 3
    
    # Align to 6h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Weekly trend filter: 1w EMA50 slope
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_rising = np.zeros_like(ema_50_1w, dtype=bool)
    ema_50_falling = np.zeros_like(ema_50_1w, dtype=bool)
    ema_50_rising[1:] = ema_50_1w[1:] > ema_50_1w[:-1]
    ema_50_falling[1:] = ema_50_1w[1:] < ema_50_1w[:-1]
    
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_50_falling)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema_50_rising_aligned[i]) or np.isnan(ema_50_falling_aligned[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above H3, weekly uptrend, volume confirmation
            long_condition = (close[i] > h3_aligned[i]) and ema_50_rising_aligned[i] and volume_confirm[i]
            # Short: price breaks below L3, weekly downtrend, volume confirmation
            short_condition = (close[i] < l3_aligned[i]) and ema_50_falling_aligned[i] and volume_confirm[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below L4 (reversal signal)
            if close[i] < l4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above H4 (reversal signal)
            if close[i] > h4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf