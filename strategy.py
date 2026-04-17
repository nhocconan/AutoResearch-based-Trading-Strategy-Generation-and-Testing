#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R Extreme + Daily Volume Spike + Choppiness Filter.
Williams %R below -80 = oversold (long setup), above -20 = overbought (short setup).
Require volume > 1.5x 20-period average for confirmation.
Use daily choppiness index > 61.8 (range regime) for mean-reversion entries.
Exit when Williams %R crosses -50 (mean reversion completion).
Timeframe: 12h for swing trading, avoids 15m/1h overtrading, captures multi-day moves.
Works in bull (buy dips) and bear (sell rallies) via mean reversion in ranging markets.
Designed for low trade frequency (<40/year) to minimize fee drag.
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
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and choppiness
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams %R on 1d (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_1d - close_1d) / (highest_high_1d - lowest_low_1d) * -100
    # Handle division by zero
    williams_r = np.where((highest_high_1d - lowest_low_1d) == 0, -50, williams_r)
    
    # Calculate volume spike on 1d (> 1.5x 20-period average)
    volume_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (volume_ma_20 * 1.5)
    
    # Calculate Choppiness Index on 1d (14-period)
    # Chop = 100 * log10(sum(ATR14) / (n * (HH14 - LL14))) / log10(n)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    hh14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_raw = np.sum(pd.Series(tr).rolling(window=14, min_periods=14).sum().values) / (14 * (hh14 - ll14))
    chop_raw = np.where((hh14 - ll14) == 0, 50, chop_raw)  # avoid division by zero
    choppiness = 100 * np.log10(chop_raw) / np.log10(14)
    # Simplified calculation: standard chop formula
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(atr_sum / (14 * (hh14 - ll14))) / np.log10(14)
    chop = np.where((hh14 - ll14) == 0, 50, chop)
    chop = np.where(np.isnan(chop), 50, chop)
    
    # Align 1d indicators to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    choppiness_aligned = align_htf_to_ltf(prices, df_1d, choppiness)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or 
            np.isnan(choppiness_aligned[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r_aligned[i]
        vol_spike = bool(volume_spike_aligned[i])
        chop = choppiness_aligned[i]
        price = close[i]
        
        if position == 0:
            # Enter long: Oversold (WR < -80) + volume spike + choppy market (Chop > 61.8)
            if wr < -80 and vol_spike and chop > 61.8:
                signals[i] = 0.25
                position = 1
            # Enter short: Overbought (WR > -20) + volume spike + choppy market (Chop > 61.8)
            elif wr > -20 and vol_spike and chop > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: WR crosses above -50 (mean reversion) OR chop drops below 38.2 (trending)
            if wr > -50 or chop < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: WR crosses below -50 (mean reversion) OR chop drops below 38.2 (trending)
            if wr < -50 or chop < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0