#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot levels with volume confirmation and choppiness regime filter
# Long when price breaks above Camarilla R3 level AND 1d volume > 1.5 * avg_volume(20) AND choppiness > 61.8 (range regime)
# Short when price breaks below Camarilla S3 level AND 1d volume > 1.5 * avg_volume(20) AND choppiness > 61.8 (range regime)
# Exit when price retests the Camarilla pivot point (midpoint of R3/S3)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Camarilla levels provide strong intraday support/resistance with high probability reactions
# Volume confirmation validates breakout strength while limiting false signals
# Choppiness filter ensures we only trade in ranging markets where mean reversion works best
# Works in both bull (buy R3 breakouts) and bear (sell S3 breakdowns) markets

name = "12h_Camarilla_R3S3_Volume_Chop"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need at least 20 completed daily bars for pivot calculation
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day)
    # Typical Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    #                  S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # But we use the more conservative R3/S3 levels
    range_1d = high_1d - low_1d
    r3 = close_1d + 1.1 * range_1d
    s3 = close_1d - 1.1 * range_1d
    pivot = (high_1d + low_1d + close_1d) / 3.0  # Standard pivot point
    
    # Align 1d Camarilla levels to 12h timeframe (wait for completed 1d bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Calculate choppiness index regime filter (14-period)
    # CHOP > 61.8 = ranging market (good for mean reversion)
    # CHOP < 38.2 = trending market (avoid for this strategy)
    atr_14 = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.append([np.nan], close[:-1]))), 
                                  np.maximum(low - low, np.abs(low - np.append([np.nan], close[:-1]))))).rolling(14, min_periods=14).mean().values
    # True range calculation simplified for performance
    tr1 = high - low
    tr2 = np.abs(high - np.append([np.nan], close[:-1]))
    tr3 = np.abs(low - np.append([np.nan], close[:-1]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(14, min_periods=14).mean().values
    
    # Choppiness = 100 * log10(sum(ATR14) / (n * ATR)) / log10(n)
    # Simplified version: CHOP = 100 * log10(atr_sum / (n * atr)) / log10(n)
    # We'll use a rolling version: CHOP = 100 * log10(rolling_sum(ATR14,14) / (14 * rolling_avg(TR,14))) / log10(14)
    tr_sum_14 = pd.Series(tr).rolling(14, min_periods=14).sum().values
    atr_sum_14 = pd.Series(atr_14).rolling(14, min_periods=14).sum().values
    chop = 100 * np.log10(tr_sum_14 / (14 * atr_sum_14)) / np.log10(14)
    chop = np.nan_to_num(chop, nan=50.0)  # Replace NaN with neutral value
    chop_regime = chop > 61.8  # Range regime
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or np.isnan(chop[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R3, volume spike, in range regime, in session
            if (close[i] > r3_aligned[i] and 
                volume_confirm[i] and 
                chop_regime[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3, volume spike, in range regime, in session
            elif (close[i] < s3_aligned[i] and 
                  volume_confirm[i] and 
                  chop_regime[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests the Camarilla pivot point
            if close[i] <= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests the Camarilla pivot point
            if close[i] >= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals