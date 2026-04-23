#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
- Long when close breaks above R3 AND close > 1d EMA34 AND volume > 2.0x 20-period average
- Short when close breaks below S3 AND close < 1d EMA34 AND volume > 2.0x 20-period average
- Exit when price retouches the 1d EMA34 (mean reversion to trend)
- Uses 1d EMA34 for HTF trend alignment and 1d Camarilla levels for structure
- Volume spike ensures institutional participation and reduces false breakouts
- Designed for both bull and bear markets: trend filter prevents counter-trend entries
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
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
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d Camarilla pivot levels (R3, S3)
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    #          S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 34 for EMA34, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter (using 1d EMA34)
        uptrend = close[i] > ema34_1d_aligned[i]
        downtrend = close[i] < ema34_1d_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Close breaks above R3 + uptrend + volume confirmation
            if close[i] > r3_aligned[i] and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S3 + downtrend + volume confirmation
            elif close[i] < s3_aligned[i] and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price retouches 1d EMA34 (mean reversion to trend)
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below EMA34
                if close[i] < ema34_1d_aligned[i] and close[i-1] >= ema34_1d_aligned[i-1]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Price crosses above EMA34
                if close[i] > ema34_1d_aligned[i] and close[i-1] <= ema34_1d_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Camarilla_R3S3_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0