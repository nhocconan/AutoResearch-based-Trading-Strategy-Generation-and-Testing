#!/usr/bin/env python3
# 6h_12h_1d_camarilla_breakout_v2
# Hypothesis: Camarilla pivot breakouts on 6h with 12h/1d trend confirmation and volume filter.
# Long when 6h close breaks above R4 with 12h EMA(21) > EMA(50) and volume > 1.5x 20-period average.
# Short when 6h close breaks below S4 with 12h EMA(21) < EMA(50) and volume > 1.5x 20-period average.
# Uses Camarilla formula: R4 = close + 1.5*(high-low), S4 = close - 1.5*(high-low) from prior 12h bar.
# Designed for 50-150 total trades over 4 years on 6h timeframe with strict entry conditions.
# Works in bull markets via upside breakouts and bear markets via downside breakdowns.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_camarilla_breakout_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for Camarilla calculation and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla: R4 = close + 1.5*(high-low), S4 = close - 1.5*(high-low)
    camarilla_r4 = close_12h + 1.5 * (high_12h - low_12h)
    camarilla_s4 = close_12h - 1.5 * (high_12h - low_12h)
    
    # Align Camarilla levels to 6h timeframe (12h -> 6h: 2 bars per 12h)
    r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # 12h EMA trend: EMA(21) > EMA(50) for uptrend, < for downtrend
    ema_21 = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_12h, ema_21)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 20  # Ensure volume MA is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vol_ma_20[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_21_aligned[i]) or np.isnan(ema_50_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: close below S4 (reversal signal)
            if close[i] < s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: close above R4 (reversal signal)
            if close[i] > r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: close above R4 with uptrend and volume surge
            if close[i] > r4_aligned[i] and ema_21_aligned[i] > ema_50_aligned[i] and vol_surge:
                position = 1
                signals[i] = 0.25
            # Short entry: close below S4 with downtrend and volume surge
            elif close[i] < s4_aligned[i] and ema_21_aligned[i] < ema_50_aligned[i] and vol_surge:
                position = -1
                signals[i] = -0.25
    
    return signals