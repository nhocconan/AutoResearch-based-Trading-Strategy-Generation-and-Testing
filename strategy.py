#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_RVI_1dTrend_Volume_Slope"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for RVI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    open_1d = df_1d['open'].values
    close_1d = df_1d['close'].values
    
    # Relative Vigor Index (RVI) calculation
    # Numerator: Close - Open
    # Denominator: High - Low
    num = close_1d - open_1d
    den = high_1d - low_1d
    
    # Avoid division by zero
    den_safe = np.where(den == 0, 1e-10, den)
    raw_rvi = num / den_safe
    
    # Smooth with 4-period SMA (standard RVI smoothing)
    rvi = pd.Series(raw_rvi).rolling(window=4, min_periods=4).mean().values
    
    # Signal line: 4-period SMA of RVI
    rvi_signal = pd.Series(rvi).rolling(window=4, min_periods=4).mean().values
    
    # RVI histogram (difference between RVI and signal)
    rvi_hist = rvi - rvi_signal
    
    # 1-day EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # Align all indicators to 4h timeframe
    rvi_aligned = align_htf_to_ltf(prices, df_1d, rvi)
    rvi_signal_aligned = align_htf_to_ltf(prices, df_1d, rvi_signal)
    rvi_hist_aligned = align_htf_to_ltf(prices, df_1d, rvi_hist)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(rvi_aligned[i]) or np.isnan(rvi_signal_aligned[i]) or 
            np.isnan(rvi_hist_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: RVI crosses above signal line + upward slope + price > EMA34 + volume
            rvi_cross_up = rvi_aligned[i] > rvi_signal_aligned[i] and rvi_aligned[i-1] <= rvi_signal_aligned[i-1]
            rvi_slope_up = rvi_aligned[i] > rvi_aligned[i-1]
            if rvi_cross_up and rvi_slope_up and close[i] > ema_34_aligned[i] and vol_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: RVI crosses below signal line + downward slope + price < EMA34 + volume
            elif (rvi_aligned[i] < rvi_signal_aligned[i] and rvi_aligned[i-1] >= rvi_signal_aligned[i-1] and
                  rvi_aligned[i] < rvi_aligned[i-1] and close[i] < ema_34_aligned[i] and vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RVI crosses below signal line or price below EMA34
            rvi_cross_down = rvi_aligned[i] < rvi_signal_aligned[i] and rvi_aligned[i-1] >= rvi_signal_aligned[i-1]
            if rvi_cross_down or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RVI crosses above signal line or price above EMA34
            rvi_cross_up = rvi_aligned[i] > rvi_signal_aligned[i] and rvi_aligned[i-1] <= rvi_signal_aligned[i-1]
            if rvi_cross_up or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals