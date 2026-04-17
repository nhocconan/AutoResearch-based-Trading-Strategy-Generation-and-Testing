#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA34 trend filter and volume spike.
Long when price breaks above Donchian upper with volume > 1.8x average and 12h EMA34 up.
Short when price breaks below Donchian lower with volume > 1.8x average and 12h EMA34 down.
Exit when price touches Donchian middle line or trend reverses.
Uses discrete position sizing (0.25) to minimize fee churn. Target: 75-200 total trades over 4 years.
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
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34
    close_12h_series = pd.Series(close_12h)
    ema_34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Donchian channels (20-period) on 4h
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        upper[i] = np.max(high[i-lookback+1:i+1])
        lower[i] = np.min(low[i-lookback+1:i+1])
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    # Volume spike: current volume > 1.8x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(50, lookback)  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or 
            np.isnan(lower[i]) or 
            np.isnan(middle[i]) or 
            np.isnan(ema_34_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        ema_12h = ema_34_12h_aligned[i]
        upper_band = upper[i]
        lower_band = lower[i]
        middle_band = middle[i]
        
        # Determine 12h trend: EMA34 slope > 0 = up, < 0 = down
        if i >= start_idx + 1:
            ema_prev = ema_34_12h_aligned[i-1]
            ema_slope = ema_12h - ema_prev
            trend_up = ema_slope > 0
            trend_down = ema_slope < 0
        else:
            trend_up = False
            trend_down = False
        
        if position == 0:
            # Long: price breaks above upper band with volume spike and 12h uptrend
            if price > upper_band and vol_spike and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band with volume spike and 12h downtrend
            elif price < lower_band and vol_spike and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price touches middle band OR 12h trend turns down
            if price <= middle_band or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price touches middle band OR 12h trend turns up
            if price >= middle_band or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_EMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0