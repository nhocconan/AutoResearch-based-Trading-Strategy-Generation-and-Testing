#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R Reversal with 1d Volume Spike and ADX Trend Filter.
Long when Williams %R < -80 (oversold) + volume > 1.5 * 20-period average volume + ADX > 25 (trending market).
Short when Williams %R > -20 (overbought) + volume > 1.5 * 20-period average volume + ADX > 25.
Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts) or volume drops below average.
Uses 1d for volume spike and ADX calculation to avoid lower-timeframe noise.
Target: 50-150 total trades over 4 years (12-37/year). Williams %R captures extreme reversals,
volume spike confirms institutional interest, ADX ensures we trade in trending conditions to avoid chop losses.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume spike and ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = (highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d) * -100
    # Handle division by zero (when high == low)
    williams_r_1d = np.where((highest_high_1d - lowest_low_1d) == 0, -50, williams_r_1d)
    
    # Calculate 1d ADX (14-period) for trend strength
    # ADX requires +DI, -DI, and DX calculation
    # +DM = High_t - High_{t-1} (if positive and > Low_{t-1} - Low_t)
    # -DM = Low_{t-1} - Low_t (if positive and > High_t - High_{t-1})
    # TR = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    # +DI = 100 * EWMA(+DM) / ATR
    # -DI = 100 * EWMA(-DM) / ATR
    # DX = 100 * |+DI - -DI| / (+DI + -DI)
    # ADX = EWMA(DX)
    
    # Calculate +DM and -DM
    high_diff = np.diff(high_1d, prepend=high_1d[0])
    low_diff = np.diff(low_1d, prepend=low_1d[0])
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # Calculate True Range (TR)
    tr1 = high_1d - low_1d
    tr2 = np.abs(np.diff(high_1d, prepend=high_1d[0]))
    tr3 = np.abs(np.diff(low_1d, prepend=low_1d[0]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR using Wilder's smoothing (EWM with alpha=1/period)
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate +DI and -DI
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d
    
    # Calculate DX and ADX
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    # Handle division by zero (when both DI are 0)
    dx_1d = np.where((plus_di_1d + minus_di_1d) == 0, 0, dx_1d)
    adx_1d = pd.Series(dx_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d volume spike (current volume > 1.5 * 20-period average volume)
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * avg_volume_1d)
    
    # Align 1d indicators to 12h timeframe
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r_1d_aligned[i]
        adx = adx_1d_aligned[i]
        vol_spike = volume_spike_1d_aligned[i] > 0.5  # boolean
        
        if position == 0:
            # Long: Oversold + volume spike + trending market
            if wr < -80 and vol_spike and adx > 25:
                signals[i] = 0.25
                position = 1
            # Short: Overbought + volume spike + trending market
            elif wr > -20 and vol_spike and adx > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R crosses above -50 (reversing from oversold) OR no volume spike
            if wr > -50 or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R crosses below -50 (reversing from overbought) OR no volume spike
            if wr < -50 or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_VolumeSpike_ADXFilter"
timeframe = "12h"
leverage = 1.0