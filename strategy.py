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
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA(50) for trend direction
    ema_50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
        ema_50_1w = ema
    
    # Align weekly EMA to daily
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Load daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily pivot points (using prior day's OHLC)
    pivot_point = np.full_like(close_1d, np.nan)
    resistance1 = np.full_like(close_1d, np.nan)
    support1 = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= 2:
        for i in range(1, len(close_1d)):
            ph = high_1d[i-1]
            pl = low_1d[i-1]
            pc = close_1d[i-1]
            
            pp = (ph + pl + pc) / 3.0
            r1 = 2 * pp - pl
            s1 = 2 * pp - ph
            
            pivot_point[i] = pp
            resistance1[i] = r1
            support1[i] = s1
    
    # Align 1d indicators to daily (no shift needed as we use prior day's data)
    pivot_point_d = align_htf_to_ltf(prices, df_1d, pivot_point)
    resistance1_d = align_htf_to_ltf(prices, df_1d, resistance1)
    support1_d = align_htf_to_ltf(prices, df_1d, support1)
    
    # Volume spike detection on daily bars
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_series = pd.Series(volume)
        vol_ma_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_point_d[i]) or 
            np.isnan(resistance1_d[i]) or
            np.isnan(support1_d[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current daily volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        # Determine trend from weekly EMA
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: Price closes above S1 with volume spike in uptrend
            if (close[i] > support1_d[i] and volume_ratio > 2.0 and uptrend):
                position = 1
                signals[i] = position_size
            # Short: Price closes below R1 with volume spike in downtrend
            elif (close[i] < resistance1_d[i] and volume_ratio > 2.0 and downtrend):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price closes below pivot OR trend changes to downtrend
            if close[i] < pivot_point_d[i] or downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price closes above pivot OR trend changes to uptrend
            if close[i] > pivot_point_d[i] or uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_Pivot_S1R1_Volume_Trend"
timeframe = "1d"
leverage = 1.0