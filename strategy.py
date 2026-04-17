#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 breakout with volume confirmation and 1w EMA50 trend filter.
Long when price breaks above R1 with volume > 1.8x 12h avg volume AND 1w EMA50 rising.
Short when price breaks below S1 with volume > 1.8x 12h avg volume AND 1w EMA50 falling.
Exit when price touches the 1w EMA50.
Uses 12h for execution and volume, 1w for EMA trend filter.
Camarilla levels derived from 1d OHLC to capture institutional pivot points.
Designed to work in both bull and bear markets by following the 1w trend with volume confirmation.
Target: 12-30 trades/year per symbol.
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
    
    # Get 12h data for execution and volume
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h volume MA (20-period)
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align 1w EMA and 12h volume MA to primary timeframe
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_50_falling)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
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
        
        # Volume confirmation: current 12h volume > 1.8x 20-bar average
        volume_confirmed = volume[i] > 1.8 * vol_ma_20_aligned[i]
        
        # Calculate Camarilla levels from 1d OHLC
        df_1d = get_htf_data(prices, '1d')
        if len(df_1d) < 1:
            signals[i] = 0.0
            continue
            
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Align 1d data to 12h timeframe
        high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
        low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        
        # Get the most recent completed 1d bar's OHLC
        period_high = high_1d_aligned[i]
        period_low = low_1d_aligned[i]
        period_close = close_1d_aligned[i]
        
        # Camarilla levels
        range_val = period_high - period_low
        if range_val <= 0:
            signals[i] = 0.0
            continue
            
        R1 = period_close + range_val * 1.1 / 12
        S1 = period_close - range_val * 1.1 / 12
        R3 = period_close + range_val * 1.1 / 4
        S3 = period_close - range_val * 1.1 / 4
        R4 = period_close + range_val * 1.1 / 2
        S4 = period_close - range_val * 1.1 / 2
        
        # Breakout conditions
        breakout_R1 = close[i] > R1
        breakout_S1 = close[i] < S1
        
        # Exit condition: touch 1w EMA50
        touch_ema = abs(close[i] - ema_50_1w[-1]) < 0.005 * close[i] if len(ema_50_1w) > 0 else False
        
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