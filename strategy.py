#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Chop
Hypothesis: Camarilla R1/S1 breakouts with 1d EMA34 trend filter, volume spike, and choppiness regime filter capture institutional breakouts with reduced false signals. 
In bull markets: price breaks above R1 (first resistance) with 1d uptrend, volume > 1.5x median, and CHOP < 61.8 (trending regime) → long. 
In bear markets: price breaks below S1 (first support) with 1d downtrend, volume spike, and CHOP < 61.8 → short. 
Uses 1d EMA34 for slower, more reliable trend and minimum holding period (3 bars) to reduce fee drag. 
Target: 75-150 trades over 4 years. Camarilla pivots from 1d provide key institutional levels that work across regimes, while chop filter avoids whipsaws in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 34:  # Need 34 for EMA and chop calculation
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: volume > 1.5x 20-period median
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (vol_median * 1.5)
    
    # Choppiness Index (CHOP) - values < 38.2 = strong trend, > 61.8 = ranging/choppy
    # We want CHOP < 61.8 to ensure we're in a trending environment
    hl_range = pd.Series(high - low)
    atr = hl_range.rolling(window=14, min_periods=14).sum()
    max_hh = pd.Series(high).rolling(window=14, min_periods=14).max()
    min_ll = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr / (max_hh - min_ll)) / np.log10(14)
    chop_values = chop.values
    chop_filter = chop_values < 61.8  # Trending regime
    
    # Load 1d data for HTF trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter (more responsive than 50, slower than 20)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous 1d bar's OHLC
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    typical_price = (h_1d + l_1d + c_1d) / 3.0
    hl_range_1d = h_1d - l_1d
    
    r1_1d = typical_price + (hl_range_1d * 1.1 / 12.0)  # R1 level
    s1_1d = typical_price - (hl_range_1d * 1.1 / 12.0)  # S1 level
    
    # Align Camarilla levels to 4h timeframe (use previous 1d bar's levels)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    bars_since_entry = 0
    
    # Start after warmup (need 34 for EMA and chop)
    start_idx = 34
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(volume_spike[i]) or 
            np.isnan(chop_filter[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        close_val = close[i]
        ema_val = ema_34_1d_aligned[i]
        r1_val = r1_1d_aligned[i]
        s1_val = s1_1d_aligned[i]
        
        # Long logic: price breaks above R1 with volume spike, 1d uptrend, and trending regime
        long_condition = (close_val > r1_val) and volume_spike[i] and (close_val > ema_val) and chop_filter[i]
        # Short logic: price breaks below S1 with volume spike, 1d downtrend, and trending regime
        short_condition = (close_val < s1_val) and volume_spike[i] and (close_val < ema_val) and chop_filter[i]
        
        # Exit logic: trend reversal or chop regime shift
        exit_long = (close_val < ema_val) or (not chop_filter[i])
        exit_short = (close_val > ema_val) or (not chop_filter[i])
        
        # Minimum holding period: 3 bars
        if position != 0 and bars_since_entry < 3:
            # Hold position regardless of signals
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            bars_since_entry = 0
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            bars_since_entry = 0
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Chop"
timeframe = "4h"
leverage = 1.0