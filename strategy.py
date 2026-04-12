#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation
    # Works in bull/bear by trading institutional levels with volume validation
    # Target: 12-30 trades/year per symbol (total 50-120 over 4 years)
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and volume context (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla pivot levels
    pivot = (high_1d[-1] + low_1d[-1] + close_1d[-1]) / 3.0
    range_1d = high_1d[-1] - low_1d[-1]
    
    # Camarilla levels for breakout
    h3 = pivot + (range_1d * 1.1 / 4)
    h4 = pivot + (range_1d * 1.1 / 2)
    l3 = pivot - (range_1d * 1.1 / 4)
    l4 = pivot - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe (constant for the day)
    h3_arr = np.full(len(df_1d), h3)
    h4_arr = np.full(len(df_1d), h4)
    l3_arr = np.full(len(df_1d), l3)
    l4_arr = np.full(len(df_1d), l4)
    
    h3_12h = align_htf_to_ltf(prices, df_1d, h3_arr)
    h4_12h = align_htf_to_ltf(prices, df_1d, h4_arr)
    l3_12h = align_htf_to_ltf(prices, df_1d, l3_arr)
    l4_12h = align_htf_to_ltf(prices, df_1d, l4_arr)
    
    # 1d volume spike filter: current volume > 1.5 * 20-day average
    vol_ma_20 = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        vol_ma_20[i] = np.mean(volume_1d[i-19:i+1])
    vol_spike = np.full(len(df_1d), np.nan)
    for i in range(19, len(df_1d)):
        if not np.isnan(vol_ma_20[i]) and vol_ma_20[i] > 0:
            vol_spike[i] = volume_1d[i] / vol_ma_20[i]
    
    vol_spike_12h = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]) or 
            np.isnan(vol_spike_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: volume spike > 1.5
        vol_confirm = vol_spike_12h[i] > 1.5
        
        # Breakout conditions
        breakout_long = close[i] > h4_12h[i]
        breakout_short = close[i] < l4_12h[i]
        
        # Entry conditions
        long_entry = breakout_long and vol_confirm
        short_entry = breakout_short and vol_confirm
        
        # Exit conditions: opposite Camarilla level touch or volume collapse
        long_exit = (close[i] < l3_12h[i]) or (vol_spike_12h[i] < 1.0)
        short_exit = (close[i] > h3_12h[i]) or (vol_spike_12h[i] < 1.0)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_camarilla_breakout_vol_v1"
timeframe = "12h"
leverage = 1.0