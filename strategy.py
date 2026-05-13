#!/usr/bin/env python3
# Hypothesis: 1d Camarilla pivot breakout with 1w trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 (1.118 * pivot) AND 1w EMA50 uptrend AND volume > 1.5x average
# Short when price breaks below Camarilla S3 (0.882 * pivot) AND 1w EMA50 downtrend AND volume > 1.5x average
# Exit when price retouches the pivot point (mean reversion to equilibrium) OR trend reversal
# Uses 1d timeframe for lower frequency, Camarilla pivots for structure, 1w EMA for trend filter, volume for confirmation.
# Target: 30-100 total trades over 4 years (7-25/year). Works in bull via breakout continuation, bear via faded rallies.

name = "1d_Camarilla_R3S3_Breakout_1wTrend_Volume_v1"
timeframe = "1d"
leverage = 1.0

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
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    # Pivot = (high + low + close) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla R3 = pivot + 1.118 * (high - low)
    r3_1d = pivot_1d + 1.118 * (high_1d - low_1d)
    # Camarilla S3 = pivot - 1.118 * (high - low)
    s3_1d = pivot_1d - 1.118 * (high_1d - low_1d)
    
    # Volume filter: current 1d volume > 1.5x 20-period average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = volume_1d > (1.5 * vol_ma_1d)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on 1w close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(pivot_1d[i]) or np.isnan(r3_1d[i]) or np.isnan(s3_1d[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_1d[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R3 AND 1w EMA50 uptrend AND volume confirmation
            if close[i] > r3_1d[i] and close[i] > ema50_1w_aligned[i] and volume_filter_1d[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S3 AND 1w EMA50 downtrend AND volume confirmation
            elif close[i] < s3_1d[i] and close[i] < ema50_1w_aligned[i] and volume_filter_1d[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price retouches pivot (mean reversion) OR trend reversal (price < 1w EMA50)
            if close[i] <= pivot_1d[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price retouches pivot (mean reversion) OR trend reversal (price > 1w EMA50)
            if close[i] >= pivot_1d[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals