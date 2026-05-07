#!/usr/bin/env python3
"""
12h_Keltner_Channel_Breakout_1dTrend_VolumeSpike
Hypothesis: Keltner Channel (20, 2*ATR) breakout on 12h timeframe with 1d EMA50 trend filter and volume spike (>2x average). Designed for low trade frequency (12-37/year) to avoid fee drag. Works in both bull and bear markets by aligning with daily trend direction. Uses discrete position sizing (0.25) to minimize churn.
"""

name = "12h_Keltner_Channel_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR for Keltner Channel (20-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Keltner Channel (20, 2*ATR)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema_20 + 2 * atr
    lower_keltner = ema_20 - 2 * atr
    
    # Volume confirmation: 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine daily trend using aligned close
        daily_close_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
        if np.isnan(daily_close_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        daily_trend_up = daily_close_aligned[i] > ema_50_1d_aligned[i]
        daily_trend_down = daily_close_aligned[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Keltner, volume spike, price above EMA50
            if (close[i] > upper_keltner[i] and 
                vol_ratio[i] > 2.0 and 
                daily_trend_up):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Keltner, volume spike, price below EMA50
            elif (close[i] < lower_keltner[i] and 
                  vol_ratio[i] > 2.0 and 
                  daily_trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below lower Keltner or trend changes
            if close[i] < lower_keltner[i] or not daily_trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above upper Keltner or trend changes
            if close[i] > upper_keltner[i] or not daily_trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals