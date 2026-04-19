#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camillo_Pivot_Squeeze_Volume_Breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot and Bollinger Bands (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Daily high, low, close for calculations
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot point (standard)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Calculate Bollinger Bands (20, 2) on daily close
    close_series_1d = pd.Series(close_1d)
    sma_20_1d = close_series_1d.rolling(window=20, min_periods=20).mean().values
    std_20_1d = close_series_1d.rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + 2 * std_20_1d
    lower_bb_1d = sma_20_1d - 2 * std_20_1d
    # Bollinger Band Width for squeeze detection
    bb_width_1d = (upper_bb_1d - lower_bb_1d) / sma_20_1d
    
    # Calculate daily ATR (14-period) for volatility filter
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.absolute(high_1d[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.absolute(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    atr_14_1d = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    # Align all daily indicators to 12h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    upper_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d)
    lower_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d)
    bb_width_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_width_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Volume confirmation: current volume > 1.8x 30-period average (12h) - moderate filter
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    # Bollinger Band squeeze signal: BB width < 20th percentile of last 50 days
    bb_width_series = pd.Series(bb_width_1d_aligned)
    bb_width_percentile_20 = bb_width_series.rolling(window=50, min_periods=20).quantile(0.20).values
    squeeze_condition = bb_width_1d_aligned < bb_width_percentile_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(upper_bb_1d_aligned[i]) or 
            np.isnan(lower_bb_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(vol_ma_30[i]) or np.isnan(bb_width_percentile_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_30[i]
        pivot = pivot_1d_aligned[i]
        upper_bb = upper_bb_1d_aligned[i]
        lower_bb = lower_bb_1d_aligned[i]
        atr = atr_14_1d_aligned[i]
        squeeze = squeeze_condition[i]
        
        volume_confirmed = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long: price breaks above upper BB during squeeze with volume
            if price > upper_bb and squeeze and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower BB during squeeze with volume
            elif price < lower_bb and squeeze and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to middle (pivot) or volatility expands
            if price < pivot or not squeeze:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to middle (pivot) or volatility expands
            if price > pivot or not squeeze:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals