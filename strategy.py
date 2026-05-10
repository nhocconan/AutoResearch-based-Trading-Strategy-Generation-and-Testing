#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla R3/S3 levels on daily chart act as institutional support/resistance.
# Breakouts above R3 or below S3 with volume confirmation and aligned daily trend filter.
# Uses 4h timeframe for balance of signal quality and trade frequency (target: 20-50 trades/year).
# Works in bull markets via breakouts above R3 in uptrend, and bear markets via breakdowns below S3 in downtrend.
# Low trade frequency expected due to multi-condition confluence.

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    typical = (high + low + close) / 3
    range_val = high - low
    S1 = close - (range_val * 1.1 / 12)
    S2 = close - (range_val * 1.1 / 6)
    S3 = close - (range_val * 1.1 / 4)
    R1 = close + (range_val * 1.1 / 12)
    R2 = close + (range_val * 1.1 / 6)
    R3 = close + (range_val * 1.1 / 4)
    return S1, S2, S3, R1, R2, R3

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    S1_1d, S2_1d, S3_1d, R1_1d, R2_1d, R3_1d = camarilla(high_1d, low_1d, close_1d)
    
    # Align Camarilla levels to 4h timeframe (wait for daily close)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    
    # Get 4h data for entry signals
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily EMA (34) + volume EMA (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(S3_1d_aligned[i]) or
            np.isnan(R3_1d_aligned[i]) or
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from daily EMA
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        uptrend = close_1d_aligned[i] > ema_34_1d_aligned[i]
        downtrend = close_1d_aligned[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 in uptrend with volume
            if close[i] > R3_1d_aligned[i] and uptrend and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 in downtrend with volume
            elif close[i] < S3_1d_aligned[i] and downtrend and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below S3 or trend changes to downtrend
            if close[i] < S3_1d_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above R3 or trend changes to uptrend
            if close[i] > R3_1d_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals