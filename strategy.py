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
    
    # Load 1d data for weekly pivot and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily EMA50 for trend
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 + ema_50_1d[i-1] * 48) / 50
    
    # Calculate weekly pivot points (using prior week's data)
    # We'll calculate weekly pivot from daily data: need to group by week
    # For simplicity, use prior day's OHLC for daily pivot (common proxy)
    pivot_point = np.full_like(close_1d, np.nan)
    resistance1 = np.full_like(close_1d, np.nan)
    support1 = np.full_like(close_1d, np.nan)
    resistance2 = np.full_like(close_1d, np.nan)
    support2 = np.full_like(close_1d, np.nan)
    resistance3 = np.full_like(close_1d, np.nan)
    support3 = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= 2:
        for i in range(1, len(close_1d)):
            # Prior day's OHLC
            ph = high_1d[i-1]
            pl = low_1d[i-1]
            pc = close_1d[i-1]
            
            pp = (ph + pl + pc) / 3.0
            r1 = 2 * pp - pl
            s1 = 2 * pp - ph
            r2 = pp + (ph - pl)
            s2 = pp - (ph - pl)
            r3 = ph + 2 * (pp - pl)
            s3 = pl - 2 * (ph - pp)
            
            pivot_point[i] = pp
            resistance1[i] = r1
            support1[i] = s1
            resistance2[i] = r2
            support2[i] = s2
            resistance3[i] = r3
            support3[i] = s3
    
    # Align 1d indicators to 6h timeframe
    ema_50_1d_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    pivot_point_6h = align_htf_to_ltf(prices, df_1d, pivot_point)
    resistance1_6h = align_htf_to_ltf(prices, df_1d, resistance1)
    support1_6h = align_htf_to_ltf(prices, df_1d, support1)
    resistance2_6h = align_htf_to_ltf(prices, df_1d, resistance2)
    support2_6h = align_htf_to_ltf(prices, df_1d, support2)
    resistance3_6h = align_htf_to_ltf(prices, df_1d, resistance3)
    support3_6h = align_htf_to_ltf(prices, df_1d, support3)
    
    # Load 6m data for entry timing (volume confirmation)
    # Since we're on 6h timeframe, we use the same data but check volume spike
    vol_ma_20 = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1d_6h[i]) or 
            np.isnan(pivot_point_6h[i]) or 
            np.isnan(resistance1_6h[i]) or 
            np.isnan(support1_6h[i]) or
            np.isnan(resistance2_6h[i]) or 
            np.isnan(support2_6h[i]) or
            np.isnan(resistance3_6h[i]) or 
            np.isnan(support3_6h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 6h volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        if position == 0:
            # Long: Price crosses above S3 with volume spike and above daily EMA50
            if (close[i] > support3_6h[i] and
                close[i] > ema_50_1d_6h[i] and
                volume_ratio > 2.0):
                position = 1
                signals[i] = position_size
            # Short: Price crosses below R3 with volume spike and below daily EMA50
            elif (close[i] < resistance3_6h[i] and
                  close[i] < ema_50_1d_6h[i] and
                  volume_ratio > 2.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price crosses below S1 or below daily EMA50
            if (close[i] < support1_6h[i] or 
                close[i] < ema_50_1d_6h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price crosses above R1 or above daily EMA50
            if (close[i] > resistance1_6h[i] or 
                close[i] > ema_50_1d_6h[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Pivot_S3R3_EMA50_Volume"
timeframe = "6h"
leverage = 1.0