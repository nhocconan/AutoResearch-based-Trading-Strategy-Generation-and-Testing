#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and chop regime filter
# Long when price breaks above Camarilla R3 AND 1d volume > 1.5 * 20-period average AND CHOP > 61.8 (range)
# Short when price breaks below Camarilla S3 AND 1d volume > 1.5 * 20-period average AND CHOP > 61.8 (range)
# Uses discrete sizing (0.25) to limit fee drag. Target: 12-25 trades/year per symbol.
# Camarilla provides structure in ranging markets; volume spike confirms breakout validity;
# Chop filter ensures we only trade in ranging conditions where mean reversion works.
# Works in bull markets via longs in range upswings and bear markets via shorts in range downswings.
# 12h timeframe reduces trade frequency to minimize fee drag while capturing medium-term range oscillations.

name = "12h_Camarilla_R3S3_1dVolumeSpike_CHOP_Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels (R3, S3) based on previous 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla levels: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    camarilla_high = close_12h + (high_12h - low_12h) * 1.1 / 4
    camarilla_low = close_12h - (high_12h - low_12h) * 1.1 / 4
    
    # Shift to use previous bar's levels (breakout of previous bar's Camarilla)
    camarilla_high = np.roll(camarilla_high, 1)
    camarilla_low = np.roll(camarilla_low, 1)
    camarilla_high[0] = np.nan  # First value invalid after roll
    camarilla_low[0] = np.nan
    
    # Align Camarilla levels to prices timeframe
    camarilla_high_aligned = align_htf_to_ltf(prices, df_12h, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_12h, camarilla_low)
    
    # Get 1d data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume spike: volume > 1.5 * 20-period average
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * vol_ma_20)
    
    # Align 1d volume spike to 12h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Calculate CHOP (choppiness index) on 12h for regime filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values  # SUM for CHOP
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    denominator = highest_high - lowest_low
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    chop = 100 * np.log10(atr_14 / denominator) / np.log10(14)
    # CHOP > 61.8 = ranging market (good for mean reversion/breakout in range)
    chop_filter = chop > 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Camarilla R3 AND 1d volume spike AND chop > 61.8 (ranging)
            if (close[i] > camarilla_high_aligned[i] and 
                volume_spike_aligned[i] > 0.5 and 
                chop_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < Camarilla S3 AND 1d volume spike AND chop > 61.8 (ranging)
            elif (close[i] < camarilla_low_aligned[i] and 
                  volume_spike_aligned[i] > 0.5 and 
                  chop_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Camarilla S3 OR chop < 38.2 (trending regime)
            if (close[i] < camarilla_low_aligned[i] or 
                chop[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Camarilla R3 OR chop < 38.2 (trending regime)
            if (close[i] > camarilla_high_aligned[i] or 
                chop[i] < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals