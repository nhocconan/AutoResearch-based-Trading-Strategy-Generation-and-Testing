#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_ChopZone_Breakout_Volume_1dTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 150:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # EMA50 for daily trend
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 14-period ATR for Choppiness Index
    atr14_1d = np.zeros(len(close_1d))
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                               np.abs(low_1d[1:] - close_1d[:-1])))
    atr14_1d[14:] = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_1d[:14] = np.nan
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # True Range for Choppiness denominator
    tr14_sum = np.zeros(len(close_1d))
    tr14_sum[14:] = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    tr14_sum[:14] = np.nan
    tr14_sum_aligned = align_htf_to_ltf(prices, df_1d, tr14_sum)
    
    # Choppiness Index (14)
    chop = np.zeros(len(close_1d))
    max_hh = np.zeros(len(close_1d))
    min_ll = np.zeros(len(close_1d))
    for i in range(len(close_1d)):
        if i >= 13:
            max_hh[i] = np.max(high_1d[i-13:i+1])
            min_ll[i] = np.min(low_1d[i-13:i+1])
        else:
            max_hh[i] = np.nan
            min_ll[i] = np.nan
    
    chop_raw = np.where((max_hh - min_ll) > 0, 
                        100 * np.log10(tr14_sum / atr14_1d / 14) / np.log10(14), 
                        50)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # Daily volume ratio for confirmation
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume_1d / vol_ma_20
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    # 4h Donchian breakout levels (20-period)
    donch_high = np.zeros(n)
    donch_low = np.zeros(n)
    for i in range(n):
        if i >= 19:
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
        else:
            donch_high[i] = np.nan
            donch_low[i] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Chop > 61.8 (range) + price breaks above Donchian high + volume spike + daily uptrend
            if (chop_aligned[i] > 61.8 and 
                close[i] > donch_high[i] and 
                vol_ratio_aligned[i] > 1.5 and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Chop > 61.8 (range) + price breaks below Donchian low + volume spike + daily downtrend
            elif (chop_aligned[i] > 61.8 and 
                  close[i] < donch_low[i] and 
                  vol_ratio_aligned[i] > 1.5 and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below Donchian low or chop drops below 38.2 (trend)
            if close[i] < donch_low[i] or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above Donchian high or chop drops below 38.2 (trend)
            if close[i] > donch_high[i] or chop_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals