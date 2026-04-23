#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla R4/S4 breakout with 1w EMA34 trend filter and volume confirmation.
Long when price breaks above R4 AND close > 1w EMA34 AND volume > 1.5x average.
Short when price breaks below S4 AND close < 1w EMA34 AND volume > 1.5x average.
Exit when price returns to the Camarilla pivot point (PP) or volume drops below average.
Camarilla levels provide precise intraday support/resistance derived from prior day.
1w EMA34 ensures trading in direction of higher timeframe trend.
Volume confirmation filters low-momentum breakouts.
Designed for 1d timeframe targeting 30-100 total trades over 4 years with low frequency to minimize fee drag.
Works in both bull and bear markets by only taking trades aligned with 1w trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Camarilla calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Load 1w data for EMA34 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA34 on 1w data
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 1d timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):  # Start from 1 to have prior day data
        # Skip if data not ready
        if np.isnan(ema34_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get prior 1d bar for Camarilla calculation
        idx_1d = i - 1  # Prior completed 1d bar
        if idx_1d < 1:
            continue
            
        # Prior day OHLC
        high_prev = high[idx_1d]
        low_prev = low[idx_1d]
        close_prev = close[idx_1d]
        
        # Calculate Camarilla levels for current day (based on prior day)
        range_prev = high_prev - low_prev
        if range_prev <= 0:
            continue
            
        pp = (high_prev + low_prev + close_prev) / 3.0
        r4 = pp + (range_prev * 1.1 / 2.0)
        s4 = pp - (range_prev * 1.1 / 2.0)
        
        price = close[i]
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]
        vol_current = volume[i]
        ema34_val = ema34_1w_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R4 AND close > 1w EMA34 AND volume spike
            if (price > r4 and close[i] > ema34_val and vol_current > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S4 AND close < 1w EMA34 AND volume spike
            elif (price < s4 and close[i] < ema34_val and vol_current > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to pivot point OR volume drops below average
                if (price <= pp or vol_current < vol_ma):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to pivot point OR volume drops below average
                if (price >= pp or vol_current < vol_ma):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Camarilla_R4_S4_1wEMA34_Volume"
timeframe = "1d"
leverage = 1.0