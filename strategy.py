#!/usr/bin/env python3
# 1D_Camarilla_Pivot_1wTrend_VolumeS
# Hypothesis: Uses weekly trend filter with daily Camarilla R3/S3 breakouts and volume confirmation on 1d timeframe.
# Targets 10-25 trades per year on 1d timeframe to minimize fee drag. Weekly trend ensures alignment with higher timeframe momentum,
# reducing false signals in choppy markets. Works in both bull and bear by following weekly trend direction.
# Focus on BTC/ETH as primary targets with strict entry conditions to avoid overtrading.

name = "1D_Camarilla_Pivot_1wTrend_VolumeS"
timeframe = "1d"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly EMA(5) for trend direction (fast EMA to capture weekly trend)
    ema_5_1w = pd.Series(df_1w['close']).ewm(span=5, adjust=False, min_periods=5).mean().values
    ema_5_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_5_1w)
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior day's OHLC
    # R3 = C + (H-L) * 1.1/4
    # S3 = C - (H-L) * 1.1/4
    camarilla_r3 = df_1d['close'] + ((df_1d['high'] - df_1d['low']) * 1.1 / 4)
    camarilla_s3 = df_1d['close'] - ((df_1d['high'] - df_1d['low']) * 1.1 / 4)
    
    # Align Camarilla levels to 1d timeframe (use prior day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    
    # Volume filter: volume > 1.5x 20-period average on 1d chart
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 5)  # Warmup for volume MA and weekly EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_5_1w_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below weekly EMA5
        price_above_ema = close[i] > ema_5_1w_aligned[i]
        price_below_ema = close[i] < ema_5_1w_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above R3 + above weekly EMA5 + volume spike
            if (close[i] > r3_aligned[i] and 
                price_above_ema and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 + below weekly EMA5 + volume spike
            elif (close[i] < s3_aligned[i] and 
                  price_below_ema and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks back below S3 (re-enters range) or volume drops below average
            if (close[i] < s3_aligned[i] or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks back above R3 (re-enters range) or volume drops below average
            if (close[i] > r3_aligned[i] or volume[i] < vol_ma[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals