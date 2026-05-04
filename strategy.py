#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R1/S1 breakout with 1w EMA50 trend filter and volume confirmation
# Uses 1w EMA50 for trend direction and Camarilla pivot levels from 1d for entry/exit
# Volume confirmation requires 1.8x average volume to ensure strong participation
# Target: 7-25 trades/year (30-100 total over 4 years) to minimize fee drag on 1d timeframe
# Works in both bull and bear markets by following the 1w trend direction and using Camarilla for structure
# Prioritizes BTC/ETH performance with SOL as secondary

name = "1d_Camarilla_R1S1_Breakout_1wEMA50_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and 1w data for trend filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 10 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels from 1d data (using completed 1d bars)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_for_pivot = df_1d['close'].values
    
    # Calculate pivot point
    pivot = (high_1d + low_1d + close_1d_for_pivot) / 3.0
    range_1d = high_1d - low_1d
    
    # Calculate Camarilla levels (R1/S1 only for tighter entries)
    camarilla_h3 = pivot + (range_1d * 1.1 / 6)  # R1
    camarilla_l3 = pivot - (range_1d * 1.1 / 6)  # S1
    
    # Align Camarilla levels to 1d timeframe (use previous completed 1d bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)  # R1
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)  # S1
    
    # Volume confirmation: 20-period EMA on 1d volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        # Camarilla breakout with 1w trend filter
        # Long: Price breaks above Camarilla H3 (R1) + volume spike + price above 1w EMA50 (uptrend)
        # Short: Price breaks below Camarilla L3 (S1) + volume spike + price below 1w EMA50 (downtrend)
        if position == 0:
            if (close[i] > camarilla_h3_aligned[i] and volume_spike and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            elif (close[i] < camarilla_l3_aligned[i] and volume_spike and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below Camarilla L3 (S1) OR price below 1w EMA50 (trend change)
            if close[i] < camarilla_l3_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above Camarilla H3 (R1) OR price above 1w EMA50 (trend change)
            if close[i] > camarilla_h3_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals