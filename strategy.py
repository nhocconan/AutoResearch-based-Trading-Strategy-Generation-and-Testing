#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Load daily data for ATR and volume filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 10-day ATR for volatility filter
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr = np.concatenate([[np.nan], tr2])
    atr10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 20-day volume average
    vol_ma20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Load 4h data for trend filter (HMA21)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    hma21_4h = calculate_hma(close_4h, 21)
    hma21_aligned = align_htf_to_ltf(prices, df_4h, hma21_4h)
    
    # Align daily indicators to 1h
    atr10_aligned = align_htf_to_ltf(prices, df_1d, atr10)
    vol_ma20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if any data is not ready
        if (np.isnan(atr10_aligned[i]) or 
            np.isnan(vol_ma20_aligned[i]) or 
            np.isnan(hma21_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        atr = atr10_aligned[i]
        vol_ma = vol_ma20_aligned[i]
        hma = hma21_aligned[i]
        
        # Volatility filter: ATR > 0 and volume > 1.5 * 20-day average
        vol_filter = volume > 1.5 * vol_ma
        
        # Trend filter: price above/below HMA21
        above_hma = price > hma
        below_hma = price < hma
        
        # Entry conditions
        if position == 0:
            # Long: price above HMA + volatility filter
            if above_hma and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: price below HMA + volatility filter
            elif below_hma and vol_filter:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses HMA or volatility drops
            exit_signal = False
            
            if position == 1:  # long position
                if price < hma or volume < 0.7 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                if price > hma or volume < 0.7 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = np.full_like(series, np.nan)
    for i in range(half_period - 1, len(series)):
        wma_half[i] = np.nansum(series[i - half_period + 1:i + 1] * np.arange(1, half_period + 1)) / (half_period * (half_period + 1) / 2)
    
    # WMA of full period
    wma_full = np.full_like(series, np.nan)
    for i in range(period - 1, len(series)):
        wma_full[i] = np.nansum(series[i - period + 1:i + 1] * np.arange(1, period + 1)) / (period * (period + 1) / 2)
    
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final WMA of raw HMA with sqrt period
    hma = np.full_like(series, np.nan)
    for i in range(sqrt_period - 1, len(series)):
        hma[i] = np.nansum(raw_hma[i - sqrt_period + 1:i + 1] * np.arange(1, sqrt_period + 1)) / (sqrt_period * (sqrt_period + 1) / 2)
    
    return hma

name = "1h_HMA21_Volatility_Filter"
timeframe = "1h"
leverage = 1.0