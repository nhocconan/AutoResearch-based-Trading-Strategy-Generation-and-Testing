#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ElderRay_BullBearPower_Volume_SMAFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA of daily close (used in Elder Ray)
    close_1d_series = pd.Series(close_1d)
    ema13_1d = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align Elder Ray to 6h timeframe (wait for daily close)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Volume filter: current volume > 1.3x 20-period average on 6h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 6h SMA50 filter for trend context
    sma50_6h = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        if np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(sma50_6h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter
        volume_ok = vol > 1.3 * vol_ma
        
        # Trend filter: price above SMA50 for long, below for short
        uptrend = price > sma50_6h[i]
        downtrend = price < sma50_6h[i]
        
        if position == 0:
            # Long: Bull power positive AND volume AND uptrend
            if bull_power_aligned[i] > 0 and volume_ok and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Bear power negative AND volume AND downtrend
            elif bear_power_aligned[i] < 0 and volume_ok and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Bull power turns negative OR volume dries up
            if bull_power_aligned[i] <= 0 or not volume_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Bear power turns positive OR volume dries up
            if bear_power_aligned[i] >= 0 or not volume_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals