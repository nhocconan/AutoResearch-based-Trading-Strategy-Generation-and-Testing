#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R4/S4 breakout with 1d EMA50 trend filter and volume confirmation
# Uses tighter Camarilla levels (R4/S4) for fewer, higher-quality breakouts
# 1d EMA50 provides smoother trend filter than EMA34
# Volume confirmation at 2.0x average to ensure strong institutional participation
# Designed for 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
# Works in both bull and bear markets by following 1d trend and using Camarilla structure

name = "12h_Camarilla_R4S4_Breakout_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from 1d data (using completed 1d bars)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_for_pivot = df_1d['close'].values
    
    # Calculate pivot point
    pivot = (high_1d + low_1d + close_1d_for_pivot) / 3.0
    range_1d = high_1d - low_1d
    
    # Calculate Camarilla levels (R4/S4 are stronger breakout levels)
    camarilla_h4 = pivot + (range_1d * 1.1 / 2)  # R4
    camarilla_l4 = pivot - (range_1d * 1.1 / 2)  # S4
    
    # Align Camarilla levels to 12h timeframe (use previous completed 1d bar)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)  # R4
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)  # S4
    
    # Volume confirmation: 20-period EMA on 12h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Camarilla breakout with 1d trend filter
        # Long: Price breaks above Camarilla H4 (R4) + volume spike + price above 1d EMA50 (uptrend)
        # Short: Price breaks below Camarilla L4 (S4) + volume spike + price below 1d EMA50 (downtrend)
        if position == 0:
            if (close[i] > camarilla_h4_aligned[i] and volume_spike and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            elif (close[i] < camarilla_l4_aligned[i] and volume_spike and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below Camarilla L4 (S4) OR price below 1d EMA50 (trend change)
            if close[i] < camarilla_l4_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above Camarilla H4 (R4) OR price above 1d EMA50 (trend change)
            if close[i] > camarilla_h4_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals