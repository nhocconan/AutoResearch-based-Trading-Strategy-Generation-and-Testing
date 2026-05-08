#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Choppiness Index regime filter with Donchian breakout and volume confirmation
# Uses Choppiness Index to identify trending vs ranging markets, enters only on Donchian breakouts
# in the direction of the trend, with volume confirmation. Avoids whipsaws in sideways markets.
# Timeframe: 12h targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by filtering for trending regimes only.

name = "12h_Chop_DonchianBreakout_Volume"
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
    
    # Get daily data once for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate daily Choppiness Index (14-period)
    # CHOP = 100 * log10(SUM(TR(14)) / (ATR(14) * 14)) / log10(14)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # True Range calculation
    tr1 = daily_high[1:] - daily_low[1:]
    tr2 = np.abs(daily_high[1:] - daily_close[:-1])
    tr3 = np.abs(daily_low[1:] - daily_close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with original index
    
    # ATR (14) - using Wilder's smoothing (equivalent to RMA)
    atr = np.full_like(daily_high, np.nan, dtype=float)
    if len(daily_high) >= 14:
        atr[13] = np.nanmean(tr[1:15])  # First ATR value
        for i in range(14, len(daily_high)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Sum of TR over 14 periods
    tr_sum = np.full_like(daily_high, np.nan, dtype=float)
    for i in range(13, len(daily_high)):
        tr_sum[i] = np.nansum(tr[i-13:i+1])
    
    # Choppiness Index
    chop = np.full_like(daily_high, np.nan, dtype=float)
    for i in range(13, len(daily_high)):
        if not np.isnan(tr_sum[i]) and not np.isnan(atr[i]) and atr[i] > 0:
            chop[i] = 100 * np.log10(tr_sum[i] / (atr[i] * 14)) / np.log10(14)
        else:
            chop[i] = np.nan
    
    # Align Choppiness Index to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Donchian Channel (20-period) on 12h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        chop_val = chop_aligned[i]
        upper_donchian = high_20[i]
        lower_donchian = low_20[i]
        vol_conf = volume_confirmed[i]
        
        if position == 0:
            # Enter long: Donchian breakout up + trending market (CHOP < 38.2) + volume confirmation
            if (not np.isnan(upper_donchian) and close[i] > upper_donchian and 
                chop_val < 38.2 and vol_conf):
                signals[i] = 0.25
                position = 1
            # Enter short: Donchian breakout down + trending market (CHOP < 38.2) + volume confirmation
            elif (not np.isnan(lower_donchian) and close[i] < lower_donchian and 
                  chop_val < 38.2 and vol_conf):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Donchian channel OR market becomes ranging (CHOP > 61.8)
            if (not np.isnan(lower_donchian) and close[i] < lower_donchian) or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Donchian channel OR market becomes ranging (CHOP > 61.8)
            if (not np.isnan(upper_donchian) and close[i] > upper_donchian) or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals