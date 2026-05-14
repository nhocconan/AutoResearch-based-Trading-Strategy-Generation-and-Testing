#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot R4/S4 breakout with 12h EMA100 trend filter and volume confirmation
# R4/S4 represent stronger Camarilla levels requiring more significant price moves, reducing false breakouts
# In bull markets: buy when price breaks above R4 with volume spike + price above 12h EMA100
# In bear markets: sell when price breaks below S4 with volume spike + price below 12h EMA100
# 12h EMA100 provides strong medium-term trend filter that adapts to both bull and bear regimes
# Volume confirmation requires 2.5x average volume to ensure institutional participation
# Target: 25-35 trades/year (100-140 total over 4 years) to minimize fee drag while maintaining edge
# Focus on BTC/ETH by requiring alignment with 12h trend and stronger breakout levels

name = "4h_Camarilla_R4S4_12hEMA100_VolumeSpike"
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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA(100) for trend filter
    close_12h = df_12h['close'].values
    ema_100_12h = pd.Series(close_12h).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_100_12h)
    
    # Get 1d data for Camarilla pivot calculation (based on prior day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formula: R4 = close + (high-low)*1.1/2, S4 = close - (high-low)*1.1/2
    camarilla_range = (high_1d - low_1d) * 1.1
    r4 = close_1d + camarilla_range / 2
    s4 = close_1d - camarilla_range / 2
    
    # Align Camarilla levels to 4h timeframe (using prior 1d bar's levels)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: 20-period EMA on 4h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_100_12h_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.5 x 20-period EMA (strict to avoid overtrading)
        volume_spike = volume[i] > (2.5 * vol_ema_20[i])
        
        # Camarilla breakout signals with 12h trend filter
        # Long: price breaks above R4 + volume spike + price above 12h EMA100
        # Short: price breaks below S4 + volume spike + price below 12h EMA100
        if position == 0:
            if (close[i] > r4_aligned[i] and volume_spike and 
                close[i] > ema_100_12h_aligned[i]):
                signals[i] = 0.30
                position = 1
            elif (close[i] < s4_aligned[i] and volume_spike and 
                  close[i] < ema_100_12h_aligned[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price breaks below S4 (reversal) OR price below 12h EMA100
            if close[i] < s4_aligned[i] or close[i] < ema_100_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price breaks above R4 (reversal) OR price above 12h EMA100
            if close[i] > r4_aligned[i] or close[i] > ema_100_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals