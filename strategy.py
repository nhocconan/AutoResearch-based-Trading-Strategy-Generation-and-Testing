#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 breakout with 1d volume confirmation and 1w EMA50 trend filter.
Long when price breaks above R1 with volume > 1.3x 1d avg volume AND 1w EMA50 rising.
Short when price breaks below S1 with volume > 1.3x 1d avg volume AND 1w EMA50 falling.
Exit when price touches the 1w EMA50.
Uses 12h for execution, 1d for volume confirmation, 1w for EMA trend filter.
Camarilla levels derived from 1d OHLC to capture institutional pivot points.
Designed to work in both bull and bear markets by following the 1w trend with volume confirmation.
Target: 12-30 trades/year per symbol (50-150 total over 4 years).
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
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50)
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_rising = ema_50_1w > np.roll(ema_50_1w, 1)
    ema_50_falling = ema_50_1w < np.roll(ema_50_1w, 1)
    ema_50_rising[0] = False
    ema_50_falling[0] = False
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume MA (20-period)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h timeframe
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_50_falling)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_rising_aligned[i]) or 
            np.isnan(ema_50_falling_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-bar average
        volume_confirmed = volume[i] > 1.3 * vol_ma_20_aligned[i]
        
        # Calculate Camarilla levels from 1d OHLC (using rolling window)
        lookback = 4  # 4x12h = 48h approximate (close to 1d)
        if i < lookback:
            signals[i] = 0.0
            continue
            
        # Get the highest high, lowest low, and last close over the lookback period
        period_high = np.max(high[i-lookback+1:i+1])
        period_low = np.min(low[i-lookback+1:i+1])
        period_close = close[i]
        
        # Camarilla levels
        range_val = period_high - period_low
        if range_val <= 0:
            signals[i] = 0.0
            continue
            
        R1 = period_close + range_val * 1.1 / 12
        S1 = period_close - range_val * 1.1 / 12
        R3 = period_close + range_val * 1.1 / 4
        S3 = period_close - range_val * 1.1 / 4
        
        # Breakout conditions
        breakout_R1 = close[i] > R1
        breakout_S1 = close[i] < S1
        
        # Exit condition: touch 1w EMA50
        ema_50_proxy = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
        touch_ema = abs(close[i] - ema_50_proxy[i]) < 0.008 * close[i]  # within 0.8%
        
        if position == 0:
            # Long: break above R1 with volume confirmation and rising 1w EMA
            if (breakout_R1 and volume_confirmed and ema_50_rising_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume confirmation and falling 1w EMA
            elif (breakout_S1 and volume_confirmed and ema_50_falling_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch 1w EMA50
            if touch_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch 1w EMA50
            if touch_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Volume_1wEMA50_Trend"
timeframe = "12h"
leverage = 1.0