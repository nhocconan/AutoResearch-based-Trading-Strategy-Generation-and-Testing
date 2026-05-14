#!/usr/bin/env python3
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
    
    # Load daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily pivot points (using prior day's OHLC)
    pivot_point = np.full_like(close_1d, np.nan)
    resistance2 = np.full_like(close_1d, np.nan)
    support2 = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= 2:
        for i in range(1, len(close_1d)):
            ph = high_1d[i-1]
            pl = low_1d[i-1]
            pc = close_1d[i-1]
            
            pp = (ph + pl + pc) / 3.0
            r2 = pp + (ph - pl)
            s2 = pp - (ph - pl)
            
            pivot_point[i] = pp
            resistance2[i] = r2
            support2[i] = s2
    
    # Align 1d indicators to 4h timeframe (48 bars per day)
    pivot_point_4h = align_htf_to_ltf(prices, df_1d, pivot_point)
    resistance2_4h = align_htf_to_ltf(prices, df_1d, resistance2)
    support2_4h = align_htf_to_ltf(prices, df_1d, support2)
    
    # Volume spike detection on 4h bars (10-period average)
    vol_ma_10 = np.full_like(volume, np.nan)
    if len(volume) >= 10:
        for i in range(9, len(volume)):
            vol_ma_10[i] = np.mean(volume[i-9:i+1])
    
    # Calculate daily volatility for dynamic thresholds
    daily_returns = np.diff(close_1d) / close_1d[:-1]
    daily_volatility = np.full_like(close_1d, np.nan)
    if len(daily_returns) >= 20:
        for i in range(19, len(daily_returns)):
            daily_volatility[i] = np.std(daily_returns[i-19:i+1])
    vol_4h = align_htf_to_ltf(prices, df_1d, daily_volatility)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_point_4h[i]) or 
            np.isnan(resistance2_4h[i]) or
            np.isnan(support2_4h[i]) or
            np.isnan(vol_ma_10[i]) or
            np.isnan(vol_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 4h volume vs 10-period average
        if vol_ma_10[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_10[i]
        
        # Dynamic volume threshold based on volatility (higher vol = lower threshold)
        vol_threshold = 1.5 + (vol_4h[i] * 10)  # Scale volatility to reasonable range
        vol_threshold = np.clip(vol_threshold, 1.5, 3.0)  # Keep between 1.5 and 3.0
        
        if position == 0:
            # Long: Price breaks above R2 with volume spike
            if (close[i] > resistance2_4h[i] and volume_ratio > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short: Price breaks below S2 with volume spike
            elif (close[i] < support2_4h[i] and volume_ratio > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price closes below pivot (mean reversion signal)
            if close[i] < pivot_point_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price closes above pivot (mean reversion signal)
            if close[i] > pivot_point_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Pivot_S2R2_VolatilityAdjusted_Volume"
timeframe = "4h"
leverage = 1.0