#!/usr/bin/env python3
# 4H_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS
# Hypothesis: Uses 12h EMA50 for trend filter (more responsive than 1d) with Camarilla R1/S1 breakouts and volume confirmation.
# Designed to capture trend-following opportunities in both bull and bear markets by using a shorter-term trend filter.
# Targets 20-40 trades per year with discrete position sizing (0.25) to minimize fee churn.

name = "4H_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS"
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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend direction
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior day's OHLC
    # R1 = C + ((H-L) * 1.1 / 12)
    # S1 = C - ((H-L) * 1.1 / 12)
    camarilla_r1 = df_1d['close'] + ((df_1d['high'] - df_1d['low']) * 1.1 / 12)
    camarilla_s1 = df_1d['close'] - ((df_1d['high'] - df_1d['low']) * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe (use prior day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1.values)
    
    # Volume filter: volume > 1.5x 30-period average on 4h chart
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 30)  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 12h EMA50
        price_above_ema = close[i] > ema_50_12h_aligned[i]
        price_below_ema = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above R1 + above 12h EMA50 + volume spike
            if (close[i] > r1_aligned[i] and 
                price_above_ema and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 + below 12h EMA50 + volume spike
            elif (close[i] < s1_aligned[i] and 
                  price_below_ema and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks back below S1 (re-enters range) or volume drops below average
            if (close[i] < s1_aligned[i] or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks back above R1 (re-enters range) or volume drops below average
            if (close[i] > r1_aligned[i] or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals