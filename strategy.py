#!/usr/bin/env python3
# 4h_donchian_12h_volume_chop_v1
# Hypothesis: 4h Donchian channel breakout with 12h trend filter, volume confirmation, and choppiness regime filter.
# Donchian(20) provides clear breakout levels, 12h HMA(50) filters trend direction (only trade with 12h trend),
# volume spike confirms institutional participation, choppiness index avoids whipsaw in ranging markets.
# Designed for 20-50 trades/year (80-200 over 4 years) to minimize fee drag.
# Works in bull/bear markets: trend filter avoids counter-trend trades, volume confirms validity, chop filter avoids ranging whipsaw.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_12h_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA for half period
    wma_half = pd.Series(series).ewm(span=half_period, adjust=False).mean().values
    # WMA for full period
    wma_full = pd.Series(series).ewm(span=period, adjust=False).mean().values
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    # Final HMA: WMA of raw_hma with sqrt_period
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean().values
    return hma

def calculate_choppiness(high, low, close, period):
    """Calculate Choppiness Index"""
    if len(close) < period:
        return np.full_like(close, np.nan, dtype=float)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Sum of True Range over period
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Highest high and lowest low over period
    hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
    ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Choppiness Index formula
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:  # Need enough for Donchian(20)
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (completed 4h candle only)
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    
    # Get 12h HTF data ONCE before loop for HMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:  # Need enough for HMA(50)
        return np.zeros(n)
    
    # Calculate 12h HMA(50) for trend filter
    close_12h = df_12h['close'].values
    hma_12h = calculate_hma(close_12h, 50)
    
    # Align 12h HMA to 4h timeframe (completed 12h candle only)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate choppiness index on 4h data
    chop = calculate_choppiness(high, low, close, 14)
    
    # Volume spike detection (20-period volume average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or 
            np.isnan(hma_12h_aligned[i]) or np.isnan(chop[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 4h Donchian lower band OR chop > 61.8 (ranging market)
            if close[i] < lower_4h_aligned[i] or chop[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 4h Donchian upper band OR chop > 61.8 (ranging market)
            if close[i] > upper_4h_aligned[i] or chop[i] > 61.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above 4h Donchian upper band, above 12h HMA (uptrend), with volume spike, chop < 61.8
            if (close[i] > upper_4h_aligned[i]) and (close[i] > hma_12h_aligned[i]) and vol_spike[i] and (chop[i] < 61.8):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below 4h Donchian lower band, below 12h HMA (downtrend), with volume spike, chop < 61.8
            elif (close[i] < lower_4h_aligned[i]) and (close[i] < hma_12h_aligned[i]) and vol_spike[i] and (chop[i] < 61.8):
                position = -1
                signals[i] = -0.25
    
    return signals