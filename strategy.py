#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d volume spike + chop regime filter
# Williams Alligator: Jaw (13-period SMMA smoothed 8), Teeth (8-period SMMA smoothed 5), Lips (5-period SMMA smoothed 3)
# Long when Lips > Teeth > Jaw (bullish alignment) AND volume > 2.0 * 20-period avg volume AND CHOP(14) > 61.8 (ranging market)
# Short when Lips < Teeth < Jaw (bearish alignment) AND volume > 2.0 * 20-period avg volume AND CHOP(14) > 61.8
# Exit when Alligator alignment breaks (Lips crosses Teeth or Jaw) OR CHOP < 50 (trending market)
# Uses discrete sizing 0.25 to balance profit potential and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Williams Alligator identifies trends, volume confirms participation, chop filter avoids false signals in strong trends

name = "12h_WilliamsAlligator_1dVolumeSpike_ChopFilter_v1"
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
    
    # Get daily data ONCE before loop for volume and chop calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams Alligator on 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    median_12h = (df_12h['high'].values + df_12h['low'].values) / 2.0
    
    # SMMA (Smoothed Moving Average) function
    def smma(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CURRENT_DATA) / PERIOD
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Alligator components: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw_raw = smma(median_12h, 13)
    teeth_raw = smma(median_12h, 8)
    lips_raw = smma(median_12h, 5)
    
    # Apply smoothing offsets: Jaw +8, Teeth +5, Lips +3
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Calculate 1d volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * avg_volume_20)
    
    # Calculate Choppiness Index on 1d timeframe
    def true_range(high, low, close_prev):
        tr1 = high - low
        tr2 = np.abs(high - close_prev)
        tr3 = np.abs(low - close_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    close_prev_1d = np.roll(close_1d, 1)
    close_prev_1d[0] = close_1d[0]
    tr = true_range(high_1d, low_1d, close_prev_1d)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # CHOP = 100 * log10(sum(ATR14) / (max(high14) - min(low14))) / log10(14)
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    
    chop = np.full_like(close_1d, np.nan, dtype=float)
    valid = (max_high_14 - min_low_14) > 0
    chop[valid] = 100 * np.log10(sum_atr_14[valid] / (max_high_14[valid] - min_low_14[valid])) / np.log10(14)
    
    # Align HTF indicators to 12h timeframe (wait for completed HTF bar)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) AND volume spike AND chop > 61.8 (ranging)
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                volume_spike_aligned[i] and chop_aligned[i] > 61.8):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) AND volume spike AND chop > 61.8 (ranging)
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                  volume_spike_aligned[i] and chop_aligned[i] > 61.8):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks OR chop < 50 (trending market)
            if not (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]) or chop_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks OR chop < 50 (trending market)
            if not (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]) or chop_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals