#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator + 12h Volume Spike + Chop Regime Filter.
Long when Jaw < Teeth < Lips (bullish alignment) AND 12h volume > 1.5x 20-period average AND Chop > 61.8 (range regime).
Short when Jaw > Teeth > Lips (bearish alignment) AND same volume/chop conditions.
Exit when Alligator alignment breaks or volume/chop conditions fail.
Uses 12h for volume filter and Chop regime, 4h for Alligator (SMAs-based Jaw/Teeth/Lips).
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for volume filter and chop regime
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 4h Williams Alligator (Jaw=TEETH=LIPS SMAs)
    # Jaw: 13-period SMA, shifted 8 bars
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA, shifted 5 bars
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA, shifted 3 bars
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate 12h volume spike filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_12h > (1.5 * vol_ma_20)
    
    # Calculate 12h Choppiness Index (CHOP)
    def calculate_chop(high, low, close, period=14):
        atr = np.zeros_like(high)
        for i in range(1, len(high)):
            atr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing for ATR
        atr_ma = np.zeros_like(atr)
        atr_ma[period] = np.mean(atr[1:period+1])
        for i in range(period+1, len(atr)):
            atr_ma[i] = (atr_ma[i-1] * (period-1) + atr[i]) / period
        
        # True range sum over period
        tr_sum = np.zeros_like(high)
        for i in range(period, len(high)):
            tr_sum[i] = np.sum(atr_ma[i-period+1:i+1])
        
        # Max high - min low over period
        max_high = np.zeros_like(high)
        min_low = np.zeros_like(low)
        for i in range(period-1, len(high)):
            max_high[i] = np.max(high[i-period+1:i+1])
            min_low[i] = np.min(low[i-period+1:i+1])
        
        # Avoid division by zero
        range_hl = max_high - min_low
        chop = np.where(range_hl > 0, 100 * np.log10(tr_sum / range_hl) / np.log10(period), 50)
        return chop
    
    chop_12h = calculate_chop(high_12h, low_12h, close_12h, 14)
    chop_range = chop_12h > 61.8  # range regime
    
    # Align 12h indicators to 4h
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike.astype(float))
    chop_range_aligned = align_htf_to_ltf(prices, df_12h, chop_range.astype(float))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_range_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment signals
        bullish_alignment = (jaw[i] < teeth[i]) and (teeth[i] < lips[i])
        bearish_alignment = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])
        
        # Volume and regime filters
        vol_ok = volume_spike_aligned[i] > 0.5
        chop_ok = chop_range_aligned[i] > 0.5
        
        if position == 0:
            # Long: Bullish Alligator alignment AND volume spike AND range regime
            if bullish_alignment and vol_ok and chop_ok:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment AND volume spike AND range regime
            elif bearish_alignment and vol_ok and chop_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator alignment breaks OR filters fail
            if not bullish_alignment or not vol_ok or not chop_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator alignment breaks OR filters fail
            if not bearish_alignment or not vol_ok or not chop_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_12hVolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0