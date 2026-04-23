#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
- Long: Price breaks above Camarilla R3 + volume > 1.5x 20-period avg + price > 1d EMA34
- Short: Price breaks below Camarilla S3 + volume > 1.5x 20-period avg + price < 1d EMA34
- Exit: Opposite Camarilla level (S3 for long, R3 for short) or EMA34 trend flip
- Uses Camarilla pivot levels for institutional support/resistance, volume for conviction, 1d EMA34 for HTF trend
- Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Camarilla breakouts work in both bull (breakout continuation) and bear (breakdown continuation) markets
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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from previous day
    # Camarilla: based on previous day's high, low, close
    # Need to resample to daily to get proper daily OHLC
    from mtf_data import get_htf_data
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for each day
    # Camarilla R3 = close + (high - low) * 1.1/4
    # Camarilla S3 = close - (high - low) * 1.1/4
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 1)  # Need 20 for volume MA, 1 for Camarilla (aligned array)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above Camarilla R3 + volume confirmation + price > 1d EMA34
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_confirm and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3 + volume confirmation + price < 1d EMA34
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_confirm and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below Camarilla S3 OR price < 1d EMA34 (trend flip)
            if (close[i] < camarilla_s3_aligned[i]) or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above Camarilla R3 OR price > 1d EMA34 (trend flip)
            if (close[i] > camarilla_r3_aligned[i]) or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeConfirm"
timeframe = "4h"
leverage = 1.0