#!/usr/bin/env python3
# 1d_Camarilla_Pivot_R3S3_Breakout_1wTrend
# Hypothesis: Use weekly Camarilla pivot levels (R3/S3) for breakout entries,
# filtered by weekly trend (EMA34) and volume spike.
# Enter long when price breaks above R3 with volume > 1.5x average and weekly EMA34 rising,
# short when price breaks below S3 with volume > 1.5x average and weekly EMA34 falling.
# Exit on opposite break (S3 for long, R3 for short) or trend reversal.
# Designed for low frequency (10-25 trades/year) to avoid fee drag.
# Camarilla levels work in both trending and ranging markets; breakouts capture momentum,
# while weekly trend filter avoids counter-trend trades. Volume spike confirms conviction.

name = "1d_Camarilla_Pivot_R3S3_Breakout_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """
    Calculate Camarilla pivot levels for given high, low, close.
    Returns R3, R2, R1, PP, S1, S2, S3.
    """
    typical = (high + low + close) / 3
    range_ = high - low
    R3 = close + range_ * 1.1 / 2
    R2 = close + range_ * 1.1 / 4
    R1 = close + range_ * 1.1 / 6
    PP = typical
    S1 = close - range_ * 1.1 / 6
    S2 = close - range_ * 1.1 / 4
    S3 = close - range_ * 1.1 / 2
    return R3, R2, R1, PP, S1, S2, S3

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla levels and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly Camarilla levels
    R3_1w, R2_1w, R1_1w, PP_1w, S1_1w, S2_1w, S3_1w = calculate_camarilla(high_1w, low_1w, close_1w)
    
    # Calculate weekly EMA34 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate average volume (20-period) for volume spike filter
    volume_series = pd.Series(volume)
    vol_avg_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Align weekly data to daily timeframe
    R3_1w_aligned = align_htf_to_ltf(prices, df_1w, R3_1w)
    S3_1w_aligned = align_htf_to_ltf(prices, df_1w, S3_1w)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1w, pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure EMA34 and volume average are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(R3_1w_aligned[i]) or np.isnan(S3_1w_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_avg_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x average
        volume_spike = volume[i] > 1.5 * vol_avg_20_aligned[i]
        
        # Trend conditions
        ema_rising = ema34_1w_aligned[i] > ema34_1w_aligned[i-1]
        ema_falling = ema34_1w_aligned[i] < ema34_1w_aligned[i-1]
        
        if position == 0:
            # LONG: Price breaks above R3, volume spike, and weekly EMA rising
            if close[i] > R3_1w_aligned[i] and volume_spike and ema_rising:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3, volume spike, and weekly EMA falling
            elif close[i] < S3_1w_aligned[i] and volume_spike and ema_falling:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below S3 or weekly EMA turns falling
            if close[i] < S3_1w_aligned[i] or not ema_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 or weekly EMA turns rising
            if close[i] > R3_1w_aligned[i] or not ema_falling:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals