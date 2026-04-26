#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dEMA50_Trend_VolumeSpike_ChopFilter
Hypothesis: Combines Camarilla R3/S3 breakouts with 1d EMA50 trend filter, volume spike confirmation, and choppiness regime filter to reduce false breakouts in sideways markets. Works in both bull and bear markets by following the daily trend while using Camarilla levels for precise entries and avoiding choppy regimes. Designed for 4h timeframe with target trades: 20-50/year (80-200 total over 4 years) to stay below fee drag threshold.
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
    
    # Get 1d data for Camarilla levels and EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla R3 and S3 levels
    R3 = close_1d_prev + (high_1d - low_1d) * 1.1 / 4
    S3 = close_1d_prev - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels (no additional delay needed as they're based on completed 1d bar)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume confirmation: 2.0x average volume (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index filter (14-period) to avoid sideways markets
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr.sum() / np.log(14) / (highest_high - lowest_low)) / np.log10(14)
    # Handle division by zero and edge cases
    chop = np.where((highest_high - lowest_low) == 0, 50.0, chop)
    chop = np.nan_to_num(chop, nan=50.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1d EMA (50), volume MA (20), ATR (14)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        R3_val = R3_aligned[i]
        S3_val = S3_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        chop_val = chop[i]
        
        # Only trade in trending markets (chop < 61.8)
        in_trending_regime = chop_val < 61.8
        
        if position == 0:
            # Long: price breaks above R3 with volume confirmation, uptrend, and trending regime
            long_signal = (high_val > R3_val) and (volume_val > 2.0 * vol_ma_val) and (close_val > ema_50_1d_val) and in_trending_regime
            # Short: price breaks below S3 with volume confirmation, downtrend, and trending regime
            short_signal = (low_val < S3_val) and (volume_val > 2.0 * vol_ma_val) and (close_val < ema_50_1d_val) and in_trending_regime
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: trend reversal or choppy regime
            if close_val < ema_50_1d_val or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: trend reversal or choppy regime
            if close_val > ema_50_1d_val or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dEMA50_Trend_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0