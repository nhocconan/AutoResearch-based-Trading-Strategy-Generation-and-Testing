#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla R3/S3 breakouts filtered by 1-day trend and volume spike on 12h timeframe.
# Goes long when price breaks above R3 and 1-day trend is up (close > EMA34) with volume > 1.5x average.
# Goes short when price breaks below S3 and 1-day trend is down (close < EMA34) with volume > 1.5x average.
# Uses 1-day EMA34 for trend filter and 20-period volume average for confirmation.
# Designed to work in both bull and bear markets by aligning with higher timeframe trend.
# Targets 12-30 trades per year on 12h timeframe with position size 0.25.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend direction
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day
    # Typical Price = (High + Low + Close) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Pivot = Typical Price
    pivot = typical_price.values
    # Range = High - Low
    range_val = (df_1d['high'] - df_1d['low']).values
    # R3 = Pivot + (Range * 1.1)
    r3 = pivot + (range_val * 1.1)
    # S3 = Pivot - (Range * 1.1)
    s3 = pivot - (range_val * 1.1)
    
    # Align Camarilla levels to 12h timeframe (previous day's levels available at open)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Warmup for 1d EMA and volume average
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_avg[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 1d
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = volume[i] > (vol_avg[i] * 1.5)
        
        if position == 0:
            # Long entry: price breaks above R3 AND 1-day uptrend AND volume spike
            if close[i] > r3_aligned[i] and uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 AND 1-day downtrend AND volume spike
            elif close[i] < s3_aligned[i] and downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 OR 1-day trend turns down
            if close[i] < s3_aligned[i] or downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R3 OR 1-day trend turns up
            if close[i] > r3_aligned[i] or uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals