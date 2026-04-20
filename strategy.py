#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout with 1d Volume Spike and Chop Regime Filter
# Uses Donchian channel (20-period) breakout for entry, filtered by 1d volume spike (>2x average)
# and chop regime (Choppiness Index > 61.8 for mean-reversion mode, < 38.2 for trend-following).
# In chop regime: fade breaks of Donchian bands (mean reversion).
# In trend regime: breakout follows direction (trend following).
# Includes ATR-based stoploss via signal=0 when price moves against position by 2x ATR.
# Designed to work in both bull (trend) and bear (range/chop) markets by adapting behavior.
# Target: 80-160 total trades over 4 years (20-40/year) to balance opportunity and cost.

name = "4h_Donchian_1dVolume_ChopRegime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for volume, chop, and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d Volume for spike detection ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values  # 20-day average
    vol_ratio_1d = vol_1d / np.where(vol_ma_1d > 0, vol_ma_1d, np.nan)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === 1d Choppiness Index for regime detection ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        tr = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
        atr_1d[i] = tr if i < 14 else (atr_1d[i-1] * 13 + tr) / 14
    atr_1d[:14] = np.nan
    
    sum_tr14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr14 / (highest_high_14 - lowest_low_14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 4h Donchian Channel (20-period) ===
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # === 4h ATR for stoploss ===
    tr4 = np.zeros(n)
    for i in range(1, n):
        tr = max(
            high_4h[i] - low_4h[i],
            abs(high_4h[i] - prices['close'].iloc[i-1]),
            abs(low_4h[i] - prices['close'].iloc[i-1])
        )
        tr4[i] = tr
    tr4[0] = high_4h[0] - low_4h[0]
    atr_4h = pd.Series(tr4).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Get values
        close_val = prices['close'].iloc[i]
        high_val = high_4h[i]
        low_val = low_4h[i]
        dc_high = high_20[i]
        dc_low = low_20[i]
        vol_ratio = vol_ratio_1d_aligned[i]
        chop_val = chop_aligned[i]
        atr_val = atr_4h[i]
        
        # Skip if any value is NaN
        if (np.isnan(dc_high) or np.isnan(dc_low) or np.isnan(vol_ratio) or 
            np.isnan(chop_val) or np.isnan(atr_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Determine regime: chop > 61.8 = mean reversion, chop < 38.2 = trend following
            if chop_val > 61.8:  # Chop regime: mean reversion
                # Fade Donchian band touches
                if low_val <= dc_low and vol_ratio > 2.0:  # Touch lower band with volume spike -> long
                    signals[i] = 0.25
                    position = 1
                elif high_val >= dc_high and vol_ratio > 2.0:  # Touch upper band with volume spike -> short
                    signals[i] = -0.25
                    position = -1
            else:  # Trend regime: breakout follows direction
                # Break Donchian bands with volume spike
                if high_val > dc_high and vol_ratio > 2.0:  # Break above upper band -> long
                    signals[i] = 0.25
                    position = 1
                elif low_val < dc_low and vol_ratio > 2.0:  # Break below lower band -> short
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit conditions: reverse signal or stoploss
            exit_signal = False
            if chop_val > 61.8:  # Chop regime: exit at opposite band
                if high_val >= dc_high:
                    exit_signal = True
            else:  # Trend regime: exit on breakdown
                if low_val < dc_low:
                    exit_signal = True
            
            # Stoploss: 2x ATR against position
            if close_val < (dc_high - 2.0 * atr_val):  # Using entry approximation
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: reverse signal or stoploss
            exit_signal = False
            if chop_val > 61.8:  # Chop regime: exit at opposite band
                if low_val <= dc_low:
                    exit_signal = True
            else:  # Trend regime: exit on breakout
                if high_val > dc_high:
                    exit_signal = True
            
            # Stoploss: 2x ATR against position
            if close_val > (dc_low + 2.0 * atr_val):  # Using entry approximation
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals