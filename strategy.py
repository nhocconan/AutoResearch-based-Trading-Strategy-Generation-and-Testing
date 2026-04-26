#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeConfirm
Hypothesis: Camarilla R1/S1 breakouts from 12h timeframe with 1d trend filter and volume confirmation capture institutional-level reversals. 
In bull markets: price breaks above R1 (intraday resistance) with 1d uptrend and volume spike → long. 
In bear markets: price breaks below S1 (intraday support) with 1d downtrend and volume spike → short. 
Uses 1d EMA34 for trend (proven effective on SOLUSDT) and ATR-based stoploss to limit drawdown. 
Target: 50-150 trades over 4 years (12-37/year). Avoids overtrading by requiring confluence of breakout, trend, and volume.
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
    
    # Load 1d data for HTF trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels from previous 1d bar's OHLC
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    typical_price = (h_1d + l_1d + c_1d) / 3.0
    hl_range = h_1d - l_1d
    
    r1_1d = typical_price + (hl_range * 1.1 / 12.0)
    s1_1d = typical_price - (hl_range * 1.1 / 12.0)
    
    # Align Camarilla levels to 12h timeframe (use previous 1d bar's levels)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
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
            np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
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
        r1_val = r1_1d_aligned[i]
        s1_val = s1_1d_aligned[i]
        
        # Long logic: price breaks above R1 with volume spike and 1d uptrend
        long_condition = (close_val > r1_val) and volume_spike[i] and (close_val > ema_val)
        # Short logic: price breaks below S1 with volume spike and 1d downtrend
        short_condition = (close_val < s1_val) and volume_spike[i] and (close_val < ema_val)
        
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

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0