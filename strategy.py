#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1dATR_Trend_Volume
# Strategy: Breakout of Camarilla R3/S3 levels with 1d ATR trend filter and volume confirmation
# Long when price breaks above R3 and 1d ATR(14) > 20-period SMA of ATR (volatile/uptrend)
# Short when price breaks below S3 and same ATR condition
# Exit when price returns to H4/L4 levels or ATR condition fails
# Uses volatility expansion breakout logic which works in both bull (breakouts continue) and bear (sharp reversals)
# Designed for 4h timeframe with selective entries to minimize trade frequency

name = "4h_Camarilla_R3_S3_Breakout_1dATR_Trend_Volume"
timeframe = "4h"
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
    
    # Calculate 1d ATR(14) for volatility trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing
    atr_14 = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr_14[13] = np.nanmean(tr[1:15])  # First ATR value
        for i in range(15, len(tr)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # 20-period SMA of ATR for trend filter
    atr_sma_20 = np.full_like(atr_14, np.nan)
    for i in range(19, len(atr_14)):
        if not np.isnan(atr_14[i-19:i+1]).any():
            atr_sma_20[i] = np.mean(atr_14[i-19:i+1])
    
    atr_condition = atr_14 > atr_sma_20  # Volatility expansion = trending market
    atr_condition_aligned = align_htf_to_ltf(prices, df_1d, atr_condition)
    
    # Calculate Camarilla levels from previous day
    # Using daily high, low, close from prior day
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Get previous day's OHLC
        prev_day_idx = i - 1
        if prev_day_idx < len(high):
            ph = high[prev_day_idx]
            pl = low[prev_day_idx]
            pc = close[prev_day_idx]
            
            # Camarilla formulas
            range_val = ph - pl
            camarilla_r3[i] = pc + range_val * 1.1 / 4
            camarilla_s3[i] = pc - range_val * 1.1 / 4
            camarilla_h4[i] = pc + range_val * 1.1 / 2
            camarilla_l4[i] = pc - range_val * 1.1 / 2
    
    # Volume confirmation: current volume > 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    volume_confirm = volume > vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(atr_condition_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R3 with volume and ATR condition
            if close[i] > camarilla_r3[i] and volume_confirm[i] and atr_condition_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 with volume and ATR condition
            elif close[i] < camarilla_s3[i] and volume_confirm[i] and atr_condition_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to H4 or ATR condition fails
            if close[i] < camarilla_h4[i] or not atr_condition_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to L4 or ATR condition fails
            if close[i] > camarilla_l4[i] or not atr_condition_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals