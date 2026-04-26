#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_ChopFilter
Hypothesis: Camarilla R1/S1 breakouts from 12h price action with 1d trend filter, volume spike, and chop regime filter capture institutional breakouts with reduced false signals. Works in bull/bear: long when price breaks above R1 with 1d uptrend, volume spike, and chop<61.8 (trending); short when breaks below S1 with 1d downtrend, volume spike, and chop<61.8. Uses discrete sizing (0.25) and minimum holding period (2 bars) to minimize fee drag. Target: 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:  # Need 30 for chop calculation
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: volume > 2.0x 20-period median
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (vol_median * 2.0)
    
    # Chop regime filter (14-period) - chop < 61.8 = trending (favor breakouts)
    hl_range = np.maximum(high, low) - np.minimum(high, low)  # True range approximation
    atr_14 = pd.Series(hl_range).rolling(window=14, min_periods=14).mean().values
    # Avoid division by zero
    close_sum = pd.Series(close).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(atr_14 * 14 / close_sum) / np.log10(14)
    chop = np.where(np.isnan(chop) | (close_sum == 0), 50, chop)  # default to neutral
    chop_filter = chop < 61.8  # trending regime
    
    # Load 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Load 12h data for Camarilla pivots (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 12h bar's OHLC
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    typical_price = (h_12h + l_12h + c_12h) / 3.0
    hl_range = h_12h - l_12h
    
    r1_12h = typical_price + (hl_range * 1.1 / 4.0)  # R1
    s1_12h = typical_price - (hl_range * 1.1 / 4.0)  # S1
    
    # Align Camarilla levels to 12h timeframe (use previous 12h bar's levels)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    bars_since_entry = 0
    
    # Start after warmup (need 34 for EMA and 14 for chop)
    start_idx = 34
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1_12h_aligned[i]) or 
            np.isnan(s1_12h_aligned[i]) or 
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
        r1_val = r1_12h_aligned[i]
        s1_val = s1_12h_aligned[i]
        
        # Long logic: price breaks above R1 with volume spike, 1d uptrend, and trending regime
        long_condition = (close_val > r1_val) and volume_spike[i] and (close_val > ema_val) and chop_filter[i]
        # Short logic: price breaks below S1 with volume spike, 1d downtrend, and trending regime
        short_condition = (close_val < s1_val) and volume_spike[i] and (close_val < ema_val) and chop_filter[i]
        
        # Exit logic: trend reversal OR chop regime shift to ranging
        exit_long = (close_val < ema_val) or (chop_filter[i] == False)
        exit_short = (close_val > ema_val) or (chop_filter[i] == False)
        
        # Minimum holding period: 2 bars
        if position != 0 and bars_since_entry < 2:
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

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0