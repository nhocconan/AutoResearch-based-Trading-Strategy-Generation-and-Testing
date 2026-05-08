#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot reversal with 12h trend filter and volume confirmation
# Uses Camarilla pivot levels (S3/S4 for long, R3/R4 for short) from 1d timeframe.
# Enters on reversal from these levels with 12h EMA trend filter and volume spike.
# Works in bull markets (trend-following reversals) and bear markets (mean reversion at extremes).
# Target: 20-50 trades/year to minimize fee drag while maintaining edge.

name = "4h_Camarilla_R3S4_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Get 12h data for trend filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    # S4 = close - 1.5 * (high - low)
    range_1d = high_1d - low_1d
    r4 = close_1d + 1.5 * range_1d
    r3 = close_1d + 1.1 * range_1d
    s3 = close_1d - 1.1 * range_1d
    s4 = close_1d - 1.5 * range_1d
    
    # Align Camarilla levels to 4h (wait for 1d bar to close)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 12h EMA(34) for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    trend_12h_up = ema_34_12h_aligned > np.roll(ema_34_12h_aligned, 1)
    trend_12h_up = np.where(np.isnan(trend_12h_up), False, trend_12h_up)
    
    # Volume spike detection (2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after enough data for indicators
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(trend_12h_up[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: price at or below S3/S4 with volume spike and 12h uptrend
            if (close[i] <= s3_aligned[i] or close[i] <= s4_aligned[i]) and volume_spike[i] and trend_12h_up[i]:
                signals[i] = 0.25
                position = 1
            # Short setup: price at or above R3/R4 with volume spike and 12h downtrend
            elif (close[i] >= r3_aligned[i] or close[i] >= r4_aligned[i]) and volume_spike[i] and not trend_12h_up[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back above S3 or trend fails
            if close[i] >= s3_aligned[i] or not trend_12h_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back below R3 or trend fails
            if close[i] <= r3_aligned[i] or trend_12h_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals