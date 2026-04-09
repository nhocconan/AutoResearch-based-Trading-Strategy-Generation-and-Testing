#!/usr/bin/env python3
# 12h_camarilla_1d_hma_volume_v1
# Hypothesis: 12h Camarilla H3/L3 levels from 1d pivot act as strong support/resistance with volume confirmation.
# Uses 12h timeframe for lower trade frequency (12-37/year target). 1d Camarilla H3/L3 provides institutional bias levels.
# Enter long when price breaks above 1d H3 with volume spike and 12h HMA(21) confirms uptrend.
# Enter short when price breaks below 1d L3 with volume spike and 12h HMA(21) confirms downtrend.
# Exit when price crosses 12h HMA(21) in opposite direction. Works in bull/bear: Camarilla filters counter-trend fakes,
# volume spike confirms participation, HMA provides smooth trend with reduced lag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_1d_hma_volume_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA for half period using EWM as approximation
    wma_half = pd.Series(series).ewm(span=half_period, adjust=False).mean().values
    # WMA for full period
    wma_full = pd.Series(series).ewm(span=period, adjust=False).mean().values
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    # Final HMA: WMA of raw_hma with sqrt_period
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean().values
    return hma

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough for HMA(21)
        return np.zeros(n)
    
    # Calculate 12h HMA(21) for trend
    close_12h = df_12h['close'].values
    hma_12h = calculate_hma(close_12h, 21)
    
    # Align 12h HMA to 12h timeframe (completed 12h candle only)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Get 1d HTF data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla H3/L3 levels (stronger bias filter than H4/L4)
    h3_1d = pivot_1d + (range_1d * 1.1 / 4)
    l3_1d = pivot_1d - (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe (completed daily candle only)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # Volume spike detection (20-period volume average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is NaN
        if (np.isnan(hma_12h_aligned[i]) or np.isnan(h3_1d_aligned[i]) or 
            np.isnan(l3_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 12h HMA
            if close[i] < hma_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 12h HMA
            if close[i] > hma_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above 12h HMA, above 1d H3, with volume spike
            if (close[i] > hma_12h_aligned[i]) and (close[i] > h3_1d_aligned[i]) and vol_spike[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below 12h HMA, below 1d L3, with volume spike
            elif (close[i] < hma_12h_aligned[i]) and (close[i] < l3_1d_aligned[i]) and vol_spike[i]:
                position = -1
                signals[i] = -0.25
    
    return signals