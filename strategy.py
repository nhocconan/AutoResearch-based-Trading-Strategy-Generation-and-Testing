#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w trend filter and volume confirmation.
Long when price breaks above Donchian(20) high and 1w EMA34 is rising.
Short when price breaks below Donchian(20) low and 1w EMA34 is falling.
Exit on opposite Donchian breakout or when 1w EMA34 flips direction.
Uses 1w for trend, 12h for entry timing and volume.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 for trend direction
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate 12h Donchian channels (20-period)
    donchian_window = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(donchian_window - 1, n):
        upper[i] = np.max(high[i - donchian_window + 1:i + 1])
        lower[i] = np.min(low[i - donchian_window + 1:i + 1])
    
    # Calculate 12h volume spike (volume > 1.5x 20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(50, donchian_window - 1)  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or 
            np.isnan(lower[i]) or 
            np.isnan(ema34_1w_aligned[i]) or 
            i >= len(vol_spike)):
            signals[i] = 0.0
            continue
        
        # 1w trend direction (using EMA slope)
        if i >= 1:
            ema34_prev = ema34_1w_aligned[i-1]
            ema34_curr = ema34_1w_aligned[i]
            trend_up = ema34_curr > ema34_prev
            trend_down = ema34_curr < ema34_prev
        else:
            trend_up = False
            trend_down = False
        
        # Donchian breakout conditions
        breakout_up = close[i] > upper[i]
        breakout_down = close[i] < lower[i]
        
        if position == 0:
            # Long: bullish breakout + volume spike + 1w uptrend
            if breakout_up and vol_spike[i] and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout + volume spike + 1w downtrend
            elif breakout_down and vol_spike[i] and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bearish breakout OR 1w trend turns down
            if breakout_down or (not trend_up and trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bullish breakout OR 1w trend turns up
            if breakout_up or (not trend_down and trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_1wEMA34_Volume"
timeframe = "12h"
leverage = 1.0