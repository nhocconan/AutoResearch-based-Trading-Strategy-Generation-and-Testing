#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Bollinger Bands with volume confirmation and chop filter.
# Long when price touches lower BB AND volume > 1.5x average AND chop > 61.8 (ranging).
# Short when price touches upper BB AND volume > 1.5x average AND chop > 61.8.
# Exit when price crosses middle BB (20-period SMA).
# Bollinger Bands capture mean reversion in ranging markets.
# Volume confirmation ensures institutional interest.
# Chop filter avoids trending markets where mean reversion fails.
# Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Bollinger Bands, volume average, and chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough for BB(20,2) and chop(14)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Bollinger Bands (20, 2)
    sma = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper = sma + 2 * std
    lower = sma - 2 * std
    
    # Calculate average volume (20-period)
    avg_volume = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index (14)
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR (14) - smoothed TR
    atr = np.full_like(tr, np.nan)
    atr[13] = np.nanmean(tr[1:14])  # First ATR: simple average of first 14 TR
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Chop = 100 * log10(sum(TR14) / (max(HH14) - min(LL14))) / log10(14)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    
    # Align indicators to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    sma_aligned = align_htf_to_ltf(prices, df_1d, sma)
    avg_volume_aligned = align_htf_to_ltf(prices, df_1d, avg_volume)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(34, 20)  # Need BB and chop periods
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or
            np.isnan(sma_aligned[i]) or
            np.isnan(avg_volume_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_conf = volume[i] > 1.5 * avg_volume_aligned[i]
        
        # Chop filter: chop > 61.8 indicates ranging market
        ranging = chop_aligned[i] > 61.8
        
        if position == 0:
            # Look for mean reversion entries in ranging market
            # Long: price at or below lower BB AND volume confirmation AND ranging
            if (close[i] <= lower_aligned[i] and 
                volume_conf and 
                ranging):
                position = 1
                signals[i] = position_size
            # Short: price at or above upper BB AND volume confirmation AND ranging
            elif (close[i] >= upper_aligned[i] and 
                  volume_conf and 
                  ranging):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses above middle BB (SMA)
            if close[i] >= sma_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses below middle BB (SMA)
            if close[i] <= sma_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_BollingerBands_Volume_Chop_v1"
timeframe = "4h"
leverage = 1.0