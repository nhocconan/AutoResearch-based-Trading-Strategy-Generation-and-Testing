#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 Breakout with 12h EMA50 Trend and Volume Confirmation
- Camarilla pivot levels (R3/S3) provide strong intraday support/resistance
- Breakout above R3 or below S3 with volume confirmation captures momentum
- 12h EMA(50) ensures alignment with higher timeframe trend
- Designed for 4h timeframe targeting 20-50 trades/year (75-200 over 4 years) to minimize fee drag
- Works in both bull and bear markets by trading breakouts in direction of 12h trend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Camarilla: R4 = close + 1.5*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4, etc.
    # We use R3 and S3 as key levels
    daily_range = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * daily_range / 4
    camarilla_s3 = close_1d - 1.1 * daily_range / 4
    
    # Align Camarilla levels to 4h timeframe (no additional delay needed for pivot points)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA12h, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Camarilla breakout signals with trend filter and volume confirmation
        # Long: price breaks above R3 (resistance) + uptrend + volume spike
        # Short: price breaks below S3 (support) + downtrend + volume spike
        long_signal = (close[i] > camarilla_r3_aligned[i] and 
                      close[i] > ema_50_12h_aligned[i] and
                      volume[i] > 2.0 * vol_ma[i])
        
        short_signal = (close[i] < camarilla_s3_aligned[i] and 
                       close[i] < ema_50_12h_aligned[i] and
                       volume[i] > 2.0 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.30
                position = 1
            elif short_signal:
                signals[i] = -0.30
                position = -1
        else:
            # Exit conditions: trend reversal or opposite Camarilla level break
            exit_signal = False
            
            if position == 1:
                # Exit long: trend reversal or price breaks below S3 (support)
                if (close[i] < ema_50_12h_aligned[i] or 
                    close[i] < camarilla_s3_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: trend reversal or price breaks above R3 (resistance)
                if (close[i] > ema_50_12h_aligned[i] or 
                    close[i] > camarilla_r3_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0