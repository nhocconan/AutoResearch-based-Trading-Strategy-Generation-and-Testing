#!/usr/bin/env python3
# 1d_1W_KAMA_Trend_SR_Breakout
# Hypothesis: On daily timeframe, use 1-week KAMA to establish trend direction and look for breakouts above weekly resistance or below weekly support with volume confirmation. This strategy targets trend continuation in both bull and bear markets by aligning with higher timeframe momentum while using daily price action for precise entry/exit. Low trade frequency expected due to weekly structure and volume filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1W_KAMA_Trend_SR_Breakout"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for KAMA, support, resistance
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === Weekly KAMA for trend direction ===
    close_1w = df_1w['close'].values
    # Calculate Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1w, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close_1w, n=1)), axis=0)  # 10-period sum of absolute changes
    # Handle first 10 values
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Calculate KAMA
    kama = np.zeros_like(close_1w)
    kama[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close_1w[i] - kama[i-1])
    
    # === Weekly Support and Resistance (using weekly high/low channels) ===
    # 20-week high for resistance, 20-week low for support
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    resistance_20w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    support_20w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # === Daily volume ratio (current vs 20-day average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align all weekly data to daily
    kama_aligned = align_htf_to_ltf(prices, df_1w, kama)
    resistance_20w_aligned = align_htf_to_ltf(prices, df_1w, resistance_20w)
    support_20w_aligned = align_htf_to_ltf(prices, df_1w, support_20w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup for weekly indicators
        # Get values
        close_val = prices['close'].iloc[i]
        kama_val = kama_aligned[i]
        resistance_val = resistance_20w_aligned[i]
        support_val = support_20w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(kama_val) or np.isnan(resistance_val) or 
            np.isnan(support_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above weekly resistance with volume confirmation and uptrend (price > weekly KAMA)
            if (close_val > resistance_val and  # Price broke above weekly resistance
                close_val > kama_val and  # Uptrend filter: price above weekly KAMA
                vol_ratio_val > 2.0):  # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below weekly support with volume confirmation and downtrend (price < weekly KAMA)
            elif (close_val < support_val and  # Price broke below weekly support
                  close_val < kama_val and  # Downtrend filter: price below weekly KAMA
                  vol_ratio_val > 2.0):  # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below weekly KAMA or breaks below weekly support
            if close_val < kama_val or close_val < support_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above weekly KAMA or breaks above weekly resistance
            if close_val > kama_val or close_val > resistance_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals