#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 Breakout with Volume Spike and Chop Filter.
Long when price breaks above R1 with volume > 1.5x 20-period average AND chop > 61.8 (ranging market).
Short when price breaks below S1 with volume > 1.5x 20-period average AND chop > 61.8.
Exit when price returns to Camarilla pivot (PP) or chop < 38.2 (trending market).
Uses 1d for Camarilla levels (calculated from prior 1d OHLC), 12h for price/volume/chop.
Target: 50-150 total trades over 4 years (12-37/year). Camarilla breakouts work in ranging markets,
chop filter avoids false signals during strong trends, volume confirmation ensures participation.
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
    
    # Get 1d data for Camarilla pivot levels (prior day OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for prior 1d: R1, S1, PP
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    # PP = (high + low + close)/3
    rng = high_1d - low_1d
    R1 = close_1d + 1.1 * rng / 12
    S1 = close_1d - 1.1 * rng / 12
    PP = (high_1d + low_1d + close_1d) / 3
    
    # Align 1d Camarilla levels to 12h timeframe (prior day's levels)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    
    # Calculate 20-period average volume on 12h
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate Chopiness Index (CHOP) on 12h: EMA of sum(|high-low|) / EMA of max(high[-1],close[-1]) - min(low[-1],close[-1])
    # True range = max(high, close_prev) - min(low, close_prev)
    close_shift = np.roll(close, 1)
    close_shift[0] = np.nan
    tr1 = np.abs(high - close_shift)
    tr2 = np.abs(low - close_shift)
    tr = np.maximum(tr1, tr2)
    atr1 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values  # CHOP typically uses 10-period EMA
    
    # Absolute price change over period
    abs_close_ch = np.abs(np.diff(close, prepend=close[0]))
    abs_ch_ema = pd.Series(abs_close_ch).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Chop = 100 * log10(sum(abs_ch_ema)/sum(atr1)) / log10(period)
    chop = 100 * np.log10(abs_ch_ema / atr1) / np.log10(10)
    chop = np.where(abs_ch_ema > 0, chop, 50)  # avoid division by zero, set to neutral 50
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(PP_aligned[i]) or np.isnan(vol_ma20[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        r1 = R1_aligned[i]
        s1 = S1_aligned[i]
        pp = PP_aligned[i]
        vol_ma = vol_ma20[i]
        ch = chop[i]
        
        if position == 0:
            # Long: price > R1, volume > 1.5x MA, chop > 61.8 (ranging)
            if price > r1 and vol > 1.5 * vol_ma and ch > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: price < S1, volume > 1.5x MA, chop > 61.8 (ranging)
            elif price < s1 and vol > 1.5 * vol_ma and ch > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < PP (return to pivot) OR chop < 38.2 (trending market)
            if price < pp or ch < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > PP (return to pivot) OR chop < 38.2 (trending market)
            if price > pp or ch < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0