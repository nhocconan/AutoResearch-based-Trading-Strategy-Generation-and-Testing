#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d Trend Filter + Volume Spike
# Williams %R identifies overbought/oversold conditions, effective for mean reversion in ranging markets.
# 1d EMA200 trend filter ensures we trade with higher timeframe direction.
# Volume spike confirms institutional participation and reduces false signals.
# Target: 15-35 trades/year (60-140 over 4 years) to avoid fee drag.
name = "6h_WilliamsR_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # 14-period Williams %R
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # 1d EMA200 for trend filter
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d EMA200 to 6h
    ema200_1d_6h = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema200_1d_6h[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Oversold (Williams %R < -80) + above 1d EMA200 + volume spike
            if williams_r[i] < -80 and close[i] > ema200_1d_6h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Overbought (Williams %R > -20) + below 1d EMA200 + volume spike
            elif williams_r[i] > -20 and close[i] < ema200_1d_6h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R rises above -50 (momentum shift) OR price below 1d EMA200
            if williams_r[i] > -50 or close[i] < ema200_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R falls below -50 (momentum shift) OR price above 1d EMA200
            if williams_r[i] < -50 or close[i] > ema200_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals