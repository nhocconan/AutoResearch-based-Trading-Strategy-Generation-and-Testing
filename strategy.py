#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with volume and chop regime filter
# Uses 1d HTF for pivot calculation, aligned to 12h with proper delay
# Volume confirmation and choppiness index regime filter to avoid false breakouts
# Designed for both bull and bear markets by using volatility-based regime filter
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    #           S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    # We'll use R3 and S3 as primary breakout levels
    daily_range = daily_high - daily_low
    camarilla_r3 = daily_close + 1.1 * daily_range * 1.1 / 4
    camarilla_s3 = daily_close - 1.1 * daily_range * 1.1 / 4
    pivot_point = (daily_high + daily_low + daily_close) / 3.0
    
    # Calculate 1d ATR(14) for volatility regime
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr3 = np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 12h timeframe with proper delay
    camarilla_r3_12h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_12h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot_point)
    atr_14_12h = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate 12h Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR14) / (max(high,n) - min(low,n))) / log10(n)
    # CHOP > 61.8 = ranging market, CHOP < 38.2 = trending market
    atr_12h = pd.Series(high - low).ewm(span=14, adjust=False, min_periods=14).values
    atr_sum_14 = pd.Series(atr_12h).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denominator = max_high_14 - min_low_14
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)
    chop = 100 * np.log10(atr_sum_14 / chop_denominator) / np.log10(14)
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_12h[i]) or np.isnan(camarilla_s3_12h[i]) or 
            np.isnan(pivot_12h[i]) or np.isnan(atr_14_12h[i]) or 
            np.isnan(chop[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in trending markets (CHOP < 40)
        # In ranging markets (CHOP > 60), avoid breakout trades
        if chop[i] >= 40:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: volume > 1.5x average
        if volume_ratio[i] < 1.5:
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # Long: price breaks above Camarilla R3 with volume in trending market
        # Short: price breaks below Camarilla S3 with volume in trending market
        if close[i] > camarilla_r3_12h[i]:
            signals[i] = 0.25  # Long 25% position
        elif close[i] < camarilla_s3_12h[i]:
            signals[i] = -0.25  # Short 25% position
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_Volume_Chop_Filter"
timeframe = "12h"
leverage = 1.0