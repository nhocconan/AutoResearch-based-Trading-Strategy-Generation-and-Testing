#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike filter and chop regime
# Long when price breaks above Camarilla R3 AND 1d volume > 1.5 * 20-period average AND chop > 61.8 (range)
# Short when price breaks below Camarilla S3 AND 1d volume > 1.5 * 20-period average AND chop > 61.8 (range)
# Uses discrete sizing (0.25) to limit fee drag. Target: 20-40 trades/year per symbol.
# Camarilla provides structure; volume spike confirms institutional interest; chop filter ensures ranging market for mean reversion.
# Works in bull markets via longs at support/resistance and bear markets via shorts at resistance/support.
# Uses 1d for HTF volume and chop to avoid noisy lower timeframe signals.

name = "4h_Camarilla_R3S3_1dVolumeSpike_Chop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels (R3, S3) based on previous 4h bar
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Camarilla levels: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_r3 = close_4h + (high_4h - low_4h) * 1.1 / 4
    camarilla_s3 = close_4h - (high_4h - low_4h) * 1.1 / 4
    
    # Shift to use previous bar's levels (breakout of previous bar's Camarilla)
    camarilla_r3 = np.roll(camarilla_r3, 1)
    camarilla_s3 = np.roll(camarilla_s3, 1)
    camarilla_r3[0] = np.nan  # First value invalid after roll
    camarilla_s3[0] = np.nan
    
    # Align Camarilla levels to prices timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Get 1d data for volume spike and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume spike filter: volume > 1.5 * 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = vol_1d > (1.5 * vol_ma_20)
    
    # Calculate 1d chop filter: CHOP(14) > 61.8 = ranging market
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop = log10(sum(TR14)/log10(14)*ATR) * 100
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = (np.log10(sum_tr_14) / np.log10(14)) / atr_14 * 100
    chop_filter = chop > 61.8  # Range market
    
    # Align 1d filters to 4h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    chop_filter_aligned = align_htf_to_ltf(prices, df_1d, chop_filter.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_filter_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price > Camarilla R3 AND volume spike AND chop > 61.8 (range)
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_spike_aligned[i] > 0.5 and 
                chop_filter_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < Camarilla S3 AND volume spike AND chop > 61.8 (range)
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_spike_aligned[i] > 0.5 and 
                  chop_filter_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price < Camarilla S3 OR volume spike disappears OR chop < 61.8 (trending)
            if (close[i] < camarilla_s3_aligned[i] or 
                volume_spike_aligned[i] < 0.5 or 
                chop_filter_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > Camarilla R3 OR volume spike disappears OR chop < 61.8 (trending)
            if (close[i] > camarilla_r3_aligned[i] or 
                volume_spike_aligned[i] < 0.5 or 
                chop_filter_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals