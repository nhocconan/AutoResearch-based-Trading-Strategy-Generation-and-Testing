#!/usr/bin/env python3
# 12h_Camarilla_Pivot_R3S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot levels from daily timeframe provide strong support/resistance.
# Breakout above R3 or below S3 with volume confirmation and 1d trend filter.
# Designed for low frequency (15-35 trades/year) to avoid fee drag.
# Works in bull/bear via trend filter; volume avoids false breakouts.

name = "12h_Camarilla_Pivot_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """
    Calculate Camarilla pivot levels for given high, low, close arrays.
    Returns R4, R3, R2, R1, PP, S1, S2, S3, S4 arrays.
    """
    typical = (high + low + close) / 3.0
    range_ = high - low
    
    R4 = close + range_ * 1.500
    R3 = close + range_ * 1.250
    R2 = close + range_ * 1.166
    R1 = close + range_ * 1.083
    PP = typical
    S1 = close - range_ * 1.083
    S2 = close - range_ * 1.166
    S3 = close - range_ * 1.250
    S4 = close - range_ * 1.500
    
    return R4, R3, R2, R1, PP, S1, S2, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels on daily data
    R4_1d, R3_1d, R2_1d, R1_1d, PP_1d, S1_1d, S2_1d, S3_1d, S4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False).values
    
    # Align Camarilla levels and EMA to 12h timeframe
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume average (20-period) for confirmation
    volume_series = pd.Series(volume)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(R3_1d_aligned[i]) or np.isnan(S3_1d_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: price above/below EMA34 on 1d
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # LONG: Break above R3 with volume and uptrend
            if close[i] > R3_1d_aligned[i] and volume_confirmed and uptrend:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 with volume and downtrend
            elif close[i] < S3_1d_aligned[i] and volume_confirmed and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Break below S3 or trend change
            if close[i] < S3_1d_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Break above R3 or trend change
            if close[i] > R3_1d_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals