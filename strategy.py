#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 20 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1d Choppiness Index (14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Max High and Min Low over 14 periods
    max_hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if max_hh[i] == min_ll[i] or np.isnan(atr_sum[i]) or np.isnan(max_hh[i]) or np.isnan(min_ll[i]):
            chop[i] = np.nan
        else:
            chop[i] = 100 * np.log10(atr_sum[i] / (max_hh[i] - min_ll[i])) / np.log10(14)
    
    chop_align = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 1w ADX (14) for trend strength
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range and Directional Movement
    tr_w = np.maximum(high_1w - low_1w, np.maximum(np.abs(high_1w - np.roll(close_1w, 1)), np.abs(low_1w - np.roll(close_1w, 1))))
    tr_w[0] = high_1w[0] - low_1w[0]
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    
    # Smooth with Wilder's smoothing (equivalent to RMA)
    def rma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) >= period:
            result[period-1] = np.nansum(arr[:period])
            for i in range(period, len(arr)):
                result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr_w = rma(tr_w, 14)
    dm_plus_smooth = rma(dm_plus, 14)
    dm_minus_smooth = rma(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_w != 0, 100 * dm_plus_smooth / atr_w, 0)
    di_minus = np.where(atr_w != 0, 100 * dm_minus_smooth / atr_w, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = rma(dx, 14)
    
    adx_align = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate 12h Bollinger Bands (20, 2) for entry signals
    close_12h = prices['close'].values
    sma_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        chop_val = chop_align[i]
        adx_val = adx_align[i]
        upper_bb_val = upper_bb[i]
        lower_bb_val = lower_bb[i]
        
        # Skip if any value is NaN
        if (np.isnan(chop_val) or np.isnan(adx_val) or 
            np.isnan(upper_bb_val) or np.isnan(lower_bb_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Chop > 61.8 (ranging) and price touches lower BB with ADX < 25 (weak trend)
            if chop_val > 61.8 and close_val <= lower_bb_val and adx_val < 25:
                signals[i] = 0.25
                position = 1
            # Short: Chop > 61.8 (ranging) and price touches upper BB with ADX < 25 (weak trend)
            elif chop_val > 61.8 and close_val >= upper_bb_val and adx_val < 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price touches upper BB or chop drops below 38.2 (trending)
            if close_val >= upper_bb_val or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches lower BB or chop drops below 38.2 (trending)
            if close_val <= lower_bb_val or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 12h_Chop_ADX_BB_MeanReversion_V1
# Uses 1d Choppiness Index (>61.8) to identify ranging markets
# Uses 1w ADX (<25) to confirm weak trend
# Enters long at lower Bollinger Band (20,2) in ranging/weak trend conditions
# Enters short at upper Bollinger Band (20,2) in ranging/weak trend conditions
# Exits when price touches opposite BB or Chop < 38.2 (trending regime)
# Designed for 12h timeframe with ~12-37 trades/year
name = "12h_Chop_ADX_BB_MeanReversion_V1"
timeframe = "12h"
leverage = 1.0