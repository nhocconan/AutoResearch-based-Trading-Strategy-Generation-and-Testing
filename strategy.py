# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
12h_1d_1w_Camarilla_R1_S1_Breakout_Volume_Trend
Hypothesis: 12h timeframe with Camarilla R1/S1 breakouts confirmed by volume spike,
            trend filter from 1w EMA34, and 1d EMA34 for additional trend confirmation.
            Works in bull/bear by using trend filters to only trade in direction of higher timeframe trend.
            Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_Camarilla_R1_S1_Breakout_Volume_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA34 on weekly close for trend
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate EMA34 on daily close for additional trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels (using previous day's data)
    def calculate_camarilla(high_arr, low_arr, close_arr):
        n_days = len(close_arr)
        R1 = np.full(n_days, np.nan)
        S1 = np.full(n_days, np.nan)
        
        for i in range(1, n_days):
            # Use previous day's OHLC
            high_prev = high_arr[i-1]
            low_prev = low_arr[i-1]
            close_prev = close_arr[i-1]
            
            # Camarilla formulas
            R1[i] = close_prev + (high_prev - low_prev) * 1.1 / 12
            S1[i] = close_prev - (high_prev - low_prev) * 1.1 / 12
        
        return R1, S1
    
    R1, S1 = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align Camarilla levels to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Calculate volume spike indicator (volume > 2.0 * 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or np.isnan(ema_34_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume confirmation required
        vol_confirm = volume_spike[i]
        
        if position == 0:
            # Long when price breaks above R1 with volume AND both weekly and daily trends are up
            if close[i] > R1_aligned[i] and vol_confirm and close[i] > ema_34_1w_aligned[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below S1 with volume AND both weekly and daily trends are down
            elif close[i] < S1_aligned[i] and vol_confirm and close[i] < ema_34_1w_aligned[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price falls below S1 (reversal) or either trend turns down
            if close[i] < S1_aligned[i] or close[i] < ema_34_1w_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price rises above R1 (reversal) or either trend turns up
            if close[i] > R1_aligned[i] or close[i] > ema_34_1w_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals