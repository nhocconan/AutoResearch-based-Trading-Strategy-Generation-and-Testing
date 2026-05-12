#!/usr/bin/env python3
# 12h_1d_1w_Vortex_Trend_With_Volume_Confirmation
# Hypothesis: Uses Vortex Indicator on 1d timeframe to determine trend direction (VI+ > VI- = bullish, VI- > VI+ = bearish) combined with weekly volatility filter (ATR ratio) and volume confirmation on 12h timeframe.
# The Vortex Indicator captures trend strength and direction with less whipsaw than traditional moving averages.
# Volume confirmation ensures trades occur with market participation.
# Weekly ATR ratio filter avoids trading during extremely low volatility periods.
# Designed for low trade frequency (50-150 total over 4 years) to minimize fee drag.
# Works in bull/bear markets by following 1d trend while using volume and volatility filters for precise entries.

name = "12h_1d_1w_Vortex_Trend_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: >1.5x 20-period average (on 12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Daily data for Vortex Indicator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Vortex Indicator (VI) on daily timeframe
    # VM+ = |High - Prior Low|, VM- = |Low - Prior High|
    # TR = True Range = max(High-Low, |High-Previous Close|, |Low-Previous Close|)
    # VI+ = Sum(VM+ over n periods) / Sum(TR over n periods)
    # VI- = Sum(VM- over n periods) / Sum(TR over n periods)
    # Using period=14 as standard
    vortex_period = 14
    
    # Calculate VM+ and VM-
    vm_plus = np.abs(high_1d - np.roll(low_1d, 1))
    vm_minus = np.abs(low_1d - np.roll(high_1d, 1))
    # Set first values to 0 (no prior day)
    vm_plus[0] = 0
    vm_minus[0] = 0
    
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    # Calculate VI+ and VI-
    vi_plus = pd.Series(vm_plus).rolling(window=vortex_period, min_periods=vortex_period).sum().values / \
              pd.Series(tr).rolling(window=vortex_period, min_periods=vortex_period).sum().values
    vi_minus = pd.Series(vm_minus).rolling(window=vortex_period, min_periods=vortex_period).sum().values / \
               pd.Series(tr).rolling(window=vortex_period, min_periods=vortex_period).sum().values
    
    # Weekly data for ATR ratio filter (volatility regime)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ATR(14) on weekly timeframe
    atr_period = 14
    # Calculate True Range for weekly
    tr1_w = high_1w - low_1w
    tr2_w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_w = np.abs(low_1w - np.roll(close_1w, 1))
    tr_w = np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))
    tr_w[0] = tr1_w[0]
    
    atr_1w = pd.Series(tr_w).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate current ATR as percentage of average ATR (volatility regime)
    # Avoid division by zero
    atr_ma_1w = pd.Series(atr_1w).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_1w / np.where(atr_ma_1w == 0, 1, atr_ma_1w)  # Ratio of current to average ATR
    
    # Volatility filter: trade only when volatility is above 20th percentile (avoid extremely low vol)
    # Using 1.0 as threshold means trading when ATR is at least average level
    vol_filter = atr_ratio > 0.8  # Allow trading when volatility is at least 80% of average
    
    # Align all indicators to 12h timeframe
    vi_plus_aligned = align_htf_to_ltf(prices, df_1d, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_1d, vi_minus)
    vol_filter_aligned = align_htf_to_ltf(prices, df_1w, vol_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(vi_plus_aligned[i]) or
            np.isnan(vi_minus_aligned[i]) or
            np.isnan(vol_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: VI+ > VI- (bullish trend) + volume spike + volatility filter
            if (vi_plus_aligned[i] > vi_minus_aligned[i] and 
                volume_spike[i] and 
                vol_filter_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: VI- > VI+ (bearish trend) + volume spike + volatility filter
            elif (vi_minus_aligned[i] > vi_plus_aligned[i] and 
                  volume_spike[i] and 
                  vol_filter_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: VI- > VI+ (trend reversal) OR loss of volume/volatility
            if (vi_minus_aligned[i] > vi_plus_aligned[i]) or \
               not (volume_spike[i] and vol_filter_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: VI+ > VI- (trend reversal) OR loss of volume/volatility
            if (vi_plus_aligned[i] > vi_minus_aligned[i]) or \
               not (volume_spike[i] and vol_filter_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals