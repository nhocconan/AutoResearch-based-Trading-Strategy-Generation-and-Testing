#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Uses 4h EMA50 for trend direction and 1d Camarilla pivot levels for structure
# Volume confirmation requires 1.5x average volume to ensure strong participation
# Session filter (08-20 UTC) to reduce noise trades
# Target: 60-150 total trades over 4 years = 15-37/year for 1h
# Uses 1d for signal direction (Camarilla structure), 4h for trend filter, 1h for entry timing
# Works in both bull and bear markets by following higher timeframe structure

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_1dTrend_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for Camarilla calculation (structure)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from 1d data (using completed 1d bars)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Calculate Camarilla levels
    camarilla_h3 = pivot + (range_1d * 1.1 / 6)  # R1
    camarilla_l3 = pivot - (range_1d * 1.1 / 6)  # S1
    camarilla_h4 = pivot + (range_1d * 1.1 / 4)  # R2
    camarilla_l2 = pivot - (range_1d * 1.1 / 4)  # S2
    
    # Align Camarilla levels to 1h timeframe (use previous completed 1d bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)  # R1
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)  # S1
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)  # R2
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)  # S2
    
    # Volume confirmation: 20-period EMA on 1h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade during 08-20 UTC
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.5 x 20-period EMA (balanced to avoid overtrading)
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Camarilla breakout with 4h trend filter
        # Long: Price breaks above Camarilla H3 (R1) + volume spike + price above 4h EMA50 (uptrend)
        # Short: Price breaks below Camarilla L3 (S1) + volume spike + price below 4h EMA50 (downtrend)
        if position == 0:
            if (close[i] > camarilla_h3_aligned[i] and volume_spike and 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            elif (close[i] < camarilla_l3_aligned[i] and volume_spike and 
                  close[i] < ema_50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Price breaks below Camarilla L3 (S1) OR price below 4h EMA50 (trend change)
            if close[i] < camarilla_l3_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Price breaks above Camarilla H3 (R1) OR price above 4h EMA50 (trend change)
            if close[i] > camarilla_h3_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals