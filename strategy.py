#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation
# Uses 12h EMA50 for trend direction and Camarilla pivot levels from 1d for entry/exit
# Volume confirmation requires 2.0x average volume to ensure strong participation
# Target: 19-50 trades/year (75-200 total over 4 years) to minimize fee drag on 4h timeframe
# Works in both bull and bear markets by following the 12h trend direction and using Camarilla for structure
# Prioritizes BTC/ETH performance with SOL as secondary

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for Camarilla calculation
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
    
    # Align Camarilla levels to 4h timeframe (use previous completed 1d bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)  # R1
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)  # S1
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)  # R2
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)  # S2
    
    # Volume confirmation: 20-period EMA on 4h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Camarilla breakout with 12h trend filter
        # Long: Price breaks above Camarilla H3 (R1) + volume spike + price above 12h EMA50 (uptrend)
        # Short: Price breaks below Camarilla L3 (S1) + volume spike + price below 12h EMA50 (downtrend)
        if position == 0:
            if (close[i] > camarilla_h3_aligned[i] and volume_spike and 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            elif (close[i] < camarilla_l3_aligned[i] and volume_spike and 
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below Camarilla L3 (S1) OR price below 12h EMA50 (trend change)
            if close[i] < camarilla_l3_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above Camarilla H3 (R1) OR price above 12h EMA50 (trend change)
            if close[i] > camarilla_h3_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals