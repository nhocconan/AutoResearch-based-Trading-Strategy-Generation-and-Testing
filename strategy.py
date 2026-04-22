#!/usr/bin/env python3
"""
Hypothesis: 12-hour Camarilla P1/S1 breakout with 1-day EMA34 trend and volume spike.
Long when price breaks above P1 with 1-day EMA34 rising and volume spike.
Short when price breaks below S1 with 1-day EMA34 falling and volume spike.
Exit when price retests pivot point (PP).
Camarilla pivot levels provide intraday support/resistance; 1-day EMA34 filters trend direction;
volume spike confirms institutional participation. Designed for low trade frequency by requiring
multiple confirmations and using 12h-level pivot levels. Works in both bull and bear markets
by following the daily trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for Camarilla pivot levels - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # PP = (H + L + C) / 3
    # P1 = C + (H - L) * 1.1 / 6
    # S1 = C - (H - L) * 1.1 / 6
    # We need previous day's H, L, C to calculate today's levels
    # Since we're using daily data, we shift by 1 to get previous day's values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot levels using previous day's data
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    p1_1d = close_1d + (high_1d - low_1d) * 1.1 / 6.0
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 6.0
    
    # Align to 12h timeframe (each day's levels apply to the entire next day)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    p1_1d_aligned = align_htf_to_ltf(prices, df_1d, p1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 1-day EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after enough data for EMA34
        # Skip if data not ready
        if (np.isnan(pp_1d_aligned[i]) or np.isnan(p1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above P1 with 1-day EMA34 rising and volume spike
            if (close[i] > p1_1d_aligned[i] and 
                ema34_1d_aligned[i] > ema34_1d_aligned[i-1] and vol_spike):
                signals[i] = 0.30
                position = 1
            # Short: Price breaks below S1 with 1-day EMA34 falling and volume spike
            elif (close[i] < s1_1d_aligned[i] and 
                  ema34_1d_aligned[i] < ema34_1d_aligned[i-1] and vol_spike):
                signals[i] = -0.30
                position = -1
        else:
            # Exit: Price retests pivot point (PP)
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below PP
                if close[i] < pp_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above PP
                if close[i] > pp_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "12H_Camarilla_P1_S1_Breakout_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0