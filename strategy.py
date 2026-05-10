#!/usr/bin/env python3
# 12h_Camarilla_Pivot_Breakout_1dTrend_VolumeFilter
# Hypothesis: Camarilla pivot levels from daily timeframe provide institutional support/resistance.
# Breakout above R3 or below S3 with 1d trend alignment and volume confirmation captures
# institutional flow. Works in bull (breakouts continue) and bear (breakdowns continue) markets.
# Target: 15-30 trades/year on 12h timeframe to minimize fee drag.

name = "12h_Camarilla_Pivot_Breakout_1dTrend_VolumeFilter"
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
    
    # Get daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # R4 = C + ((H-L) * 1.5000), R3 = C + ((H-L) * 1.2500), etc.
    # S4 = C - ((H-L) * 1.5000), S3 = C - ((H-L) * 1.2500), etc.
    # We use previous day's H, L, C to calculate today's levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Avoid NaN from shift
    valid = ~(np.isnan(prev_high) | np.isnan(prev_low) | np.isnan(prev_close))
    
    # Calculate Camarilla levels
    H_L = prev_high - prev_low
    R3 = prev_close + (H_L * 1.2500)
    S3 = prev_close - (H_L * 1.2500)
    
    # Align Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Daily EMA for trend filter (34-period)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (24-period MA on 12h chart = 12 days)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 24 for volume MA, and valid Camarilla levels
    start_idx = 24
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Daily trend filter
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: price breaks above R3 + daily uptrend + volume spike
            if close[i] > R3_aligned[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 + daily downtrend + volume spike
            elif close[i] < S3_aligned[i] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below S3 or daily trend turns down
            if close[i] < S3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above R3 or daily trend turns up
            if close[i] > R3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals