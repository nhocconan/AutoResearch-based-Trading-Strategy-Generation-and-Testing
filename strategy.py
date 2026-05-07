#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Squeeze
# Hypothesis: 4h chart strategy using Camarilla R1/S1 breakouts with 1d EMA34 trend filter, volume confirmation, and Bollinger Band squeeze release to capture breakouts from low volatility.
# Designed to avoid false breakouts in sideways markets by requiring Bollinger Band width to be at or below 20th percentile before breakout (volatility contraction followed by expansion).
# Target: 25-40 trades/year per symbol to minimize fee decay while maintaining edge in both bull and bear markets.

timeframe = "4h"
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Squeeze"
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
    
    # Get 1d data for EMA trend filter and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d closes for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Bollinger Bands (20, 2) on 1d closes
    sma_20_1d = pd.Series(df_1d['close'].values).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(df_1d['close'].values).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + 2 * std_20_1d
    lower_bb_1d = sma_20_1d - 2 * std_20_1d
    bb_width_1d = (upper_bb_1d - lower_bb_1d) / sma_20_1d  # Normalized width
    
    # Align Bollinger Band width to 4h timeframe
    bb_width_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_width_1d)
    
    # Calculate 20th percentile of BB width for squeeze condition (using expanding window)
    bb_width_percentile = np.full_like(bb_width_1d_aligned, np.nan)
    for i in range(len(bb_width_1d_aligned)):
        if i < 20:
            bb_width_percentile[i] = np.nan
        else:
            bb_width_percentile[i] = np.percentile(bb_width_1d_aligned[:i+1], 20)
    
    # Squeeze condition: BB width at or below 20th percentile (low volatility)
    bb_width_squeeze = bb_width_1d_aligned <= bb_width_percentile
    
    # Get daily data for Camarilla levels
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    camarilla_r1 = d_close + 1.1 * (d_high - d_low) / 12
    camarilla_s1 = d_close - 1.1 * (d_high - d_low) / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume spike detection: 1.5x average volume (6-period = 1 day on 4h chart)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 6)  # Ensure we have EMA, BB, and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(bb_width_1d_aligned[i]) or np.isnan(bb_width_percentile[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Entry condition: breakout from Bollinger Band squeeze with volume and trend confirmation
        squeeze_release = not bb_width_squeeze[i-1] and bb_width_squeeze[i]  # Just exited squeeze
        
        if position == 0:
            # Long: close > R1 with volume squeeze release, volume spike, price above 1d EMA34
            if (close[i] > camarilla_r1_aligned[i] and 
                squeeze_release and 
                volume[i] > 1.5 * vol_ma[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: close < S1 with volume squeeze release, volume spike, price below 1d EMA34
            elif (close[i] < camarilla_s1_aligned[i] and 
                  squeeze_release and 
                  volume[i] > 1.5 * vol_ma[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: touch S1 (opposite level) or trend failure (price below 1d EMA34)
            if close[i] < camarilla_s1_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: touch R1 (opposite level) or trend failure (price above 1d EMA34)
            if close[i] > camarilla_r1_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals