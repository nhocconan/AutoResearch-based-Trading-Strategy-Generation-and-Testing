#!/usr/bin/env python3
"""
6h_Camarilla_R4S4_Breakout_12hTrend_VolumeConfirm
Hypothesis: Camarilla R4/S4 breakouts with 12h trend filter and volume confirmation capture strong momentum moves. 
In bull markets: price breaks above R4 (strong resistance) with 12h uptrend and volume spike → long. 
In bear markets: price breaks below S4 (strong support) with 12h downtrend and volume spike → short. 
Uses 12h EMA50 for trend (more responsive than daily) and minimum holding period (2 bars) to reduce fee drag. 
Target: 50-150 trades over 4 years. Camarilla pivots from 12h provide institutional levels that work across regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:  # Need 20 for volume median
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: volume > 2.0x 20-period median
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (vol_median * 2.0)
    
    # Load 12h data for HTF trend filter and Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla pivot levels from previous 12h bar's OHLC
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    typical_price = (h_12h + l_12h + c_12h) / 3.0
    hl_range = h_12h - l_12h
    
    r4_12h = typical_price + (hl_range * 1.1 / 2.0)
    s4_12h = typical_price - (hl_range * 1.1 / 2.0)
    
    # Align Camarilla levels to 6h timeframe (use previous 12h bar's levels)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    bars_since_entry = 0
    
    # Start after warmup (need 50 for EMA)
    start_idx = 50
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(r4_12h_aligned[i]) or 
            np.isnan(s4_12h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        close_val = close[i]
        ema_val = ema_50_12h_aligned[i]
        r4_val = r4_12h_aligned[i]
        s4_val = s4_12h_aligned[i]
        
        # Long logic: price breaks above R4 with volume spike and 12h uptrend
        long_condition = (close_val > r4_val) and volume_spike[i] and (close_val > ema_val)
        # Short logic: price breaks below S4 with volume spike and 12h downtrend
        short_condition = (close_val < s4_val) and volume_spike[i] and (close_val < ema_val)
        
        # Exit logic: trend reversal
        exit_long = close_val < ema_val
        exit_short = close_val > ema_val
        
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

name = "6h_Camarilla_R4S4_Breakout_12hTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0