#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 Breakout with 1d EMA34 Trend Filter and Volume Spike
- Long: Close breaks above R3 (Camarilla resistance) + price > 1d EMA34 (uptrend) + volume > 2.0x 20-period average
- Short: Close breaks below S3 (Camarilla support) + price < 1d EMA34 (downtrend) + volume > 2.0x 20-period average
- Exit: Close retreats below R3 (for longs) or above S3 (for shorts)
- Uses Camarilla levels from daily timeframe for structure, 4h for execution
- Volume spike confirms institutional participation
- Discrete position sizing (0.25) to minimize fee churn
- Target: 20-50 trades/year (80-200 over 4 years) to avoid fee drag
- Camarilla pivot levels work in both bull and bear markets by identifying key reversal zones
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4, etc.
    # Actually: R3 = close + 1.1*(high-low)*1.1/6, S3 = close - 1.1*(high-low)*1.1/6
    # Simplified: R3 = close + 1.1*(high-low)*1.1/6, S3 = close - 1.1*(high-low)*1.1/6
    # Correct formula: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    camarilla_range = df_1d['high'] - df_1d['low']
    r3 = df_1d['close'] + camarilla_range * 1.1 / 4
    s3 = df_1d['close'] - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 needs 34, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1d EMA34
        uptrend = close[i] > ema34_aligned[i]
        downtrend = close[i] < ema34_aligned[i]
        
        # Camarilla breakout signals with trend filter and volume confirmation
        # Long: Close breaks above R3 + uptrend + volume spike
        # Short: Close breaks below S3 + downtrend + volume spike
        long_signal = (close[i] > r3_aligned[i] and 
                      uptrend and
                      volume[i] > 2.0 * vol_ma[i])
        
        short_signal = (close[i] < s3_aligned[i] and 
                       downtrend and
                       volume[i] > 2.0 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Close retreats below R3 (for longs) or above S3 (for shorts)
            exit_signal = False
            
            if position == 1:
                # Exit long: Close moves back below R3 level
                if close[i] <= r3_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Close moves back above S3 level
                if close[i] >= s3_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0