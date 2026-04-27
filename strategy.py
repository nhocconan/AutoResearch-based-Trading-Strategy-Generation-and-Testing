#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_Regime
Hypothesis: Camarilla R1/S1 breakout on 4h with 12h EMA50 trend filter, volume spike, and choppiness regime filter.
Designed for 4h timeframe targeting 75-200 total trades over 4 years.
Uses discrete position sizing (0.25) to minimize fee churn.
In trending regimes (price > EMA50 for longs, < EMA50 for shorts) and low chop (CHOP < 61.8),
breakouts at R1/S1 with volume spike capture momentum continuations.
Exit on trend reversal (close crosses EMA50) or high chop (CHOP > 61.8).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla (previous completed day)
    df_1d = get_htf_data(prices, '1d')
    
    # Camarilla levels from previous 1d bar (completed)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    rng = prev_high - prev_low
    r1 = prev_close + (rng * 1.1 / 12)
    s1 = prev_close - (rng * 1.1 / 12)
    
    # Align Camarilla levels to 4h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Get 12h data for EMA50 and choppiness filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA50 trend filter
    ema_50 = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # 12h Choppiness Index (CHOP) regime filter
    # CHOP = 100 * log10(sum(ATR over period) / log10(highest_high - lowest_low))
    atr_period = 14
    chop_period = 14
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate True Range
    tr1 = np.abs(high_12h[1:] - low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # ATR calculation
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Sum of ATR over chop_period
    atr_sum = pd.Series(atr).rolling(window=chop_period, min_periods=chop_period).sum().values
    
    # Highest high and lowest low over chop_period
    highest_high = pd.Series(high_12h).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low = pd.Series(low_12h).rolling(window=chop_period, min_periods=chop_period).min().values
    
    # Choppiness Index
    chop = np.zeros_like(close_12h)
    mask = (highest_high - lowest_low) > 0
    chop[mask] = 100 * np.log10(atr_sum[mask] / np.log10(highest_high[mask] - lowest_low[mask]))
    chop[~mask] = 50.0  # neutral when range is zero
    
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # Volume spike: current > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25  # Discrete size to reduce fee churn
    
    # Warmup: need 1d shift, EMA50, ATR, vol avg, chop
    start_idx = max(30, 50, atr_period + chop_period, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_50_aligned[i]
        chop_val = chop_aligned[i]
        vol_spike = volume_spike[i]
        
        # Regime filter: only trade in low chop (trending market)
        low_chop = chop_val < 61.8
        
        if position == 0:
            # Look for entry: Camarilla R1/S1 breakout with EMA alignment, volume spike, and low chop
            long_condition = (close_val > r1_val and 
                            close_val > ema_val and 
                            vol_spike and 
                            low_chop)
            short_condition = (close_val < s1_val and 
                             close_val < ema_val and 
                             vol_spike and 
                             low_chop)
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Exit long: price crosses below EMA50 (trend reversal) OR high chop (range market)
            if close_val < ema_val or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above EMA50 (trend reversal) OR high chop (range market)
            if close_val > ema_val or chop_val > 61.8:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0