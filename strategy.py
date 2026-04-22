# Hypothesis: 12h Camarilla Pivot R4/S4 breakout with 1d EMA34 trend filter and volume spike confirmation.  
# Uses only 3 conditions: price breaks R4/S4, volume > 2x 20-bar avg, and price >/< 1d EMA34.  
# Designed to capture strong breakouts in both bull (long) and bear (short) markets with low trade frequency.  
# Target: 12-37 trades/year to avoid fee drag.

#!/usr/bin/env python3
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
    
    # Load 12h data for entry signal alignment (used in exit logic)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Load 1d data for pivot points and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close for Camarilla pivot points (use prior bar to avoid look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (R4/S4 are breakout levels)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_ = high_1d - low_1d
    r4 = close_1d + range_ * 1.1 / 2  # Resistance level 4
    s4 = close_1d - range_ * 1.1 / 2  # Support level 4
    
    # Align R4/S4 to 12h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1d EMA34 for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R4 with volume spike AND above 1d EMA34 (uptrend)
            if (close[i] > r4_aligned[i] and volume[i] > 2.0 * vol_avg_20[i] and 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S4 with volume spike AND below 1d EMA34 (downtrend)
            elif (close[i] < s4_aligned[i] and volume[i] > 2.0 * vol_avg_20[i] and 
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Hold position: maintain signal until exit condition
            if position == 1:
                signals[i] = 0.25
            else:  # position == -1
                signals[i] = -0.25
            
            # Exit: Price crosses back to opposite S1/R1 level (tighter reversion level)
            if position == 1:
                # Exit long: Price closes below S1 (calculated from same day's pivot)
                s1 = close_1d[i] - (high_1d[i] - low_1d[i]) * 1.1 / 12
                if close[i] < s1:
                    signals[i] = 0.0
                    position = 0
            else:  # position == -1
                # Exit short: Price closes above R1 (calculated from same day's pivot)
                r1 = close_1d[i] + (high_1d[i] - low_1d[i]) * 1.1 / 12
                if close[i] > r1:
                    signals[i] = 0.0
                    position = 0
    
    return signals

name = "12H_Camarilla_R4_S4_Breakout_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0