#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for 12h analysis
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12-period ATR on daily data
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12 = pd.Series(tr).rolling(window=12, min_periods=12).mean().values
    
    # Align daily ATR to 12h timeframe
    atr_12_aligned = align_htf_to_ltf(prices, df_1d, atr_12)
    
    # Calculate 50-period EMA on daily data for trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: current volume > 2 * 12-period average (12h * 12 = 144h ~ 6 days)
    volume_ma12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need EMA and ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma12[i]) or 
            np.isnan(atr_12_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 12-period average
        volume_filter = volume[i] > (2.0 * volume_ma12[i])
        # Trend filter: price above/below EMA50
        trend_up = close[i] > ema_50_aligned[i]
        trend_down = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Long: price above EMA50 with volume confirmation (trend continuation)
            if trend_up and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: price below EMA50 with volume confirmation (trend continuation)
            elif trend_down and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below EMA50 or volume drops
            if not trend_up or not volume_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above EMA50 or volume drops
            if not trend_down or not volume_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_EMA50_VolumeTrend_Filter_v1"
timeframe = "12h"
leverage = 1.0