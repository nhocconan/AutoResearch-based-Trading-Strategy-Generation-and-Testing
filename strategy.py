#!/usr/bin/env python3
# 4h_Camarilla_Pivot_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Use Camarilla pivot levels from 1d for breakout entries, filtered by 1d EMA trend and volume spike.
# Enter long when price breaks above R3 with volume confirmation and 1d EMA up, short when breaks below S3 with volume confirmation and 1d EMA down.
# Exit on opposite break of S3/R3 or trend reversal. Designed for low frequency (20-50 trades/year) to avoid fee drag.
# Camarilla levels work well in ranging markets, and trend filter prevents whipsaws in trends.
# Volume spike confirms institutional interest at key levels.

name = "4h_Camarilla_Pivot_R3_S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """
    Calculate Camarilla pivot levels for the day.
    Returns R4, R3, R2, R1, PP, S1, S2, S3, S4.
    """
    typical = (high + low + close) / 3
    range_ = high - low
    R4 = close + range_ * 1.1 / 2
    R3 = close + range_ * 1.1 / 4
    R2 = close + range_ * 1.1 / 6
    R1 = close + range_ * 1.1 / 12
    PP = typical
    S1 = close - range_ * 1.1 / 12
    S2 = close - range_ * 1.1 / 6
    S3 = close - range_ * 1.1 / 4
    S4 = close - range_ * 1.1 / 2
    return R4, R3, R2, R1, PP, S1, S2, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot, EMA trend, and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily Camarilla levels
    R4_1d, R3_1d, R2_1d, R1_1d, PP_1d, S1_1d, S2_1d, S3_1d, S4_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Calculate daily EMA for trend filter
    ema_span = 34
    close_1d_series = pd.Series(close_1d)
    ema_1d = close_1d_series.ewm(span=ema_span, adjust=False, min_periods=ema_span).values
    
    # Calculate daily average volume for volume spike filter
    vol_ma_span = 20
    vol_1d_series = pd.Series(volume_1d)
    vol_ma_1d = vol_1d_series.rolling(window=vol_ma_span, min_periods=vol_ma_span).mean().values
    
    # Align all 1d data to 4h timeframe
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure EMA and volume MA are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(R3_1d_aligned[i]) or np.isnan(S3_1d_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume spike: current volume > 1.5 * 20-day average volume (aligned)
        volume_spike = volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        # Trend filter: EMA rising or falling
        ema_rising = ema_1d_aligned[i] > ema_1d_aligned[i-1]
        ema_falling = ema_1d_aligned[i] < ema_1d_aligned[i-1]
        
        if position == 0:
            # LONG: Price breaks above R3 with volume spike and EMA rising
            if close[i] > R3_1d_aligned[i] and volume_spike and ema_rising:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with volume spike and EMA falling
            elif close[i] < S3_1d_aligned[i] and volume_spike and ema_falling:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below S3 or EMA turns down
            if close[i] < S3_1d_aligned[i] or not ema_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 or EMA turns up
            if close[i] > R3_1d_aligned[i] or not ema_falling:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals