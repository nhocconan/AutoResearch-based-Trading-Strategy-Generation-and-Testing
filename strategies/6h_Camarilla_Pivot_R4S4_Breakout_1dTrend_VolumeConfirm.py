#!/usr/bin/env python3
"""
6h_Camarilla_Pivot_R4S4_Breakout_1dTrend_VolumeConfirm
Hypothesis: Camarilla R4/S4 breakouts with daily trend filter and volume confirmation capture strong momentum moves in both bull and bear markets. 
In bull markets: price breaks above R4 (strong resistance) with daily uptrend and volume spike → long. 
In bear markets: price breaks below S4 (strong support) with daily downtrend and volume spike → short. 
Uses discrete sizing (0.25) and minimum holding period (2 bars) to reduce fee drag. 
Target: 50-150 trades over 4 years. Camarilla pivots from 1d provide institutional levels that work across regimes.
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
    
    # Load daily data for HTF trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous day's OHLC
    # Camarilla: R4 = C + ((H-L) * 1.1/2), S4 = C - ((H-L) * 1.1/2)
    # where C = (H+L+Close)/3 (typical price)
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    typical_price = (h_1d + l_1d + c_1d) / 3.0
    hl_range = h_1d - l_1d
    
    r4_1d = typical_price + (hl_range * 1.1 / 2.0)
    s4_1d = typical_price - (hl_range * 1.1 / 2.0)
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    bars_since_entry = 0
    
    # Start after warmup (need 34 for EMA)
    start_idx = 34
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r4_1d_aligned[i]) or 
            np.isnan(s4_1d_aligned[i]) or 
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
        ema_val = ema_34_1d_aligned[i]
        r4_val = r4_1d_aligned[i]
        s4_val = s4_1d_aligned[i]
        
        # Long logic: price breaks above R4 with volume spike and daily uptrend
        long_condition = (close_val > r4_val) and volume_spike[i] and (close_val > ema_val)
        # Short logic: price breaks below S4 with volume spike and daily downtrend
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

name = "6h_Camarilla_Pivot_R4S4_Breakout_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0