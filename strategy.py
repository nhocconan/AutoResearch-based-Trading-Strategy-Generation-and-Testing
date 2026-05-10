#!/usr/bin/env python3
# 1H_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
# Hypothesis: Combines daily trend filter (EMA50) with 1h chart breakouts at Camarilla R1/S1 levels and volume confirmation.
# Daily EMA50 provides robust trend direction that works in both bull and bear markets, while 1h timeframe allows precise entry timing.
# Targets 15-35 trades per year (~60-140 over 4 years) by using strict entry conditions and session filter (08-20 UTC).

name = "1H_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "1h"
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
    
    # Get daily data for EMA trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from prior day's OHLC
    # R1 = C + ((H-L) * 1.1 / 12)
    # S1 = C - ((H-L) * 1.1 / 12)
    camarilla_r1 = df_1d['close'] + ((df_1d['high'] - df_1d['low']) * 1.1 / 12)
    camarilla_s1 = df_1d['close'] - ((df_1d['high'] - df_1d['low']) * 1.1 / 12)
    
    # Align Camarilla levels to 1h timeframe (use prior day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1.values)
    
    # Volume filter: volume > 1.8x 50-period average on 1h chart
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_threshold = vol_ma * 1.8
    
    # Session filter: 08-20 UTC (pre-compute hours)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 1)  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not session_mask[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below daily EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above R1 + above daily EMA50 + volume spike + session
            if (close[i] > r1_aligned[i] and 
                price_above_ema and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below S1 + below daily EMA50 + volume spike + session
            elif (close[i] < s1_aligned[i] and 
                  price_below_ema and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks back below S1 (re-enters range) or volume drops below average
            if (close[i] < s1_aligned[i] or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks back above R1 (re-enters range) or volume drops below average
            if (close[i] > r1_aligned[i] or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals