# 4H_CAMARILLA_PIVOT_VOLUME_TREND
# Hypothesis: Camarilla R3/S3 level breaks with volume confirmation and 12h EMA trend filter
# Targets: 20-40 trades/year per symbol, works in bull/bear via trend filter
# Uses 4h primary timeframe with 12h trend filter to reduce whipsaw

#!/usr/bin/env python3
name = "4H_Camarilla_R3S3_Breakout_12hTrend_Volume"
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
    
    # 12h trend filter: EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_12h = close_12h > ema50_12h
    trend_up_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_up_12h)
    
    # Daily Camarilla levels (using prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3 and S3 levels
    camarilla_range = high_1d - low_1d
    r3 = close_1d + camarilla_range * 1.1 / 4
    s3 = close_1d - camarilla_range * 1.1 / 4
    
    # Align to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.8 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(trend_up_12h_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 + uptrend + volume
            if close[i] > r3_aligned[i] and trend_up_12h_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 + downtrend + volume
            elif close[i] < s3_aligned[i] and not trend_up_12h_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below S3 OR trend turns down
            if close[i] < s3_aligned[i] or not trend_up_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above R3 OR trend turns up
            if close[i] > r3_aligned[i] or trend_up_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals