#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily HTF data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate weekly pivot levels from daily data (Friday close)
    # Use Friday's data for weekly pivot (week starts Monday, ends Friday)
    # We'll use the last available daily data as proxy for weekly
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align all daily indicators to 6h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # Main timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20) > 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or
            np.isnan(vol_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        high_i = high[i]
        low_i = low[i]
        atr_val = atr_1d_aligned[i]
        weekly_pivot_val = weekly_pivot_aligned[i]
        weekly_r1_val = weekly_r1_aligned[i]
        weekly_s1_val = weekly_s1_aligned[i]
        vol_ok = vol_filter[i]
        
        # Volatility filter: only trade when ATR > 0
        vol_filter_ok = atr_val > 0
        
        if position == 0:
            # Long: price breaks above weekly S1 with volume and volatility (bullish continuation)
            if high_i > weekly_s1_val and vol_ok and vol_filter_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly R1 with volume and volatility (bearish reversal)
            elif low_i < weekly_r1_val and vol_ok and vol_filter_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below weekly pivot OR volatility drops
            if low_i < weekly_pivot_val or not vol_filter_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above weekly pivot OR volatility drops
            if high_i > weekly_pivot_val or not vol_filter_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_S1R1_Breakout_VolumeFilter"
timeframe = "6h"
leverage = 1.0