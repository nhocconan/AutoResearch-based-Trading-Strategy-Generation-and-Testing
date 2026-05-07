#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: On 4h chart, buy when price breaks above Camarilla R3 level with daily uptrend filter and volume confirmation,
# sell when price breaks below Camarilla S3 level with daily downtrend filter and volume confirmation.
# Uses daily EMA34 trend filter to capture multi-day momentum while avoiding false breakouts in ranging markets.
# Designed for low trade frequency (~20-50/year) to minimize fee drag and work in both bull and bear markets.
# Camarilla levels derived from prior day's range provide institutional pivot points with statistical edge.
timeframe = "4h"
name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
leverage = 1.0

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
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Daily EMA trend filter (34-period)
    ema_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla levels from previous day's range
    # We need previous day's high, low, close - aligned to current 4h bar
    ph = df_1d['high'].shift(1).values  # Previous day's high
    pl = df_1d['low'].shift(1).values   # Previous day's low
    pc = df_1d['close'].shift(1).values # Previous day's close
    
    # Align to 4h timeframe
    ph_aligned = align_htf_to_ltf(prices, df_1d, ph)
    pl_aligned = align_htf_to_ltf(prices, df_1d, pl)
    pc_aligned = align_htf_to_ltf(prices, df_1d, pc)
    
    # Calculate Camarilla levels
    range_val = ph_aligned - pl_aligned
    r3 = pc_aligned + (range_val * 1.1 / 2)  # R3 = C + (H-L)*1.1/2
    s3 = pc_aligned - (range_val * 1.1 / 2)  # S3 = C - (H-L)*1.1/2
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Wait for warmup
        # Skip if any critical value is NaN
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(ph_aligned[i]) or np.isnan(pl_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 + daily uptrend + volume spike
            if close[i] > r3[i] and ema_1d_aligned[i] > ema_1d_aligned[i-1] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + daily downtrend + volume spike
            elif close[i] < s3[i] and ema_1d_aligned[i] < ema_1d_aligned[i-1] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below S3 or daily trend turns down
            if close[i] < s3[i]:
                signals[i] = 0.0
                position = 0
            elif ema_1d_aligned[i] < ema_1d_aligned[i-1]:  # Daily trend turns down
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above R3 or daily trend turns up
            if close[i] > r3[i]:
                signals[i] = 0.0
                position = 0
            elif ema_1d_aligned[i] > ema_1d_aligned[i-1]:  # Daily trend turns up
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals