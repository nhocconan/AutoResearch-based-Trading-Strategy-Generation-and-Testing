#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_CloseFilter
Hypothesis: Camarilla R1/S1 breakouts with 1d EMA34 trend filter and close-based entry capture institutional level breaks with momentum. 
In bull markets: price closes above R1 (first resistance) with 1d uptrend → long. 
In bear markets: price closes below S1 (first support) with 1d downtrend → short. 
Uses discrete position sizing (0.25) to reduce fee churn. Target: 75-150 trades over 4 years (19-38/year) on 4h timeframe. 
Requires BTC/ETH edge via 1d trend filter; avoids SOL-only bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need warmup for EMA
        return np.zeros(n)
    
    close = prices['close'].values
    
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
    
    hl_range = h_1d - l_1d
    r1_1d = c_1d + (hl_range * 1.1 / 12.0)
    s1_1d = c_1d - (hl_range * 1.1 / 12.0)
    
    # Align Camarilla levels to 4h timeframe (use previous 1d bar's levels)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 34 for EMA)
    start_idx = 34
    
    for i in range(start_idx, n):
        close_val = close[i]
        ema_val = ema_34_1d_aligned[i]
        r1_val = r1_1d_aligned[i]
        s1_val = s1_1d_aligned[i]
        
        # Skip if any data not ready
        if np.isnan(ema_val) or np.isnan(r1_val) or np.isnan(s1_val):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: price closes above R1 with 1d uptrend
        long_condition = (close_val > r1_val) and (close_val > ema_val)
        # Short logic: price closes below S1 with 1d downtrend
        short_condition = (close_val < s1_val) and (close_val < ema_val)
        
        # Exit logic: trend reversal
        exit_long = close_val < ema_val
        exit_short = close_val > ema_val
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_CloseFilter"
timeframe = "4h"
leverage = 1.0