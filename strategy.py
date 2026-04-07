#!/usr/bin/env python3
"""
12h Donchian Breakout with Daily Trend Filter and Volume Confirmation.
Long when price breaks above Donchian(20) high with daily uptrend and volume > 1.2x average.
Short when price breaks below Donchian(20) low with daily downtrend and volume > 1.2x average.
Exit when price returns to Donchian midpoint.
Uses daily timeframe for trend filter to reduce false breakouts in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_daily_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily trend filter (EMA 50) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Donchian Channel (20-period) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # === Volume confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to midpoint
            if close[i] <= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to midpoint
            if close[i] >= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume confirmation
            if vol_ratio[i] < 1.2:
                signals[i] = 0.0
                continue
            
            # Entry: Donchian breakout with daily trend filter
            if close[i] > highest_high[i] and close[i] > ema_50_1d_aligned[i]:
                # Breakout above upper band with daily uptrend -> long
                position = 1
                signals[i] = 0.25
            elif close[i] < lowest_low[i] and close[i] < ema_50_1d_aligned[i]:
                # Breakdown below lower band with daily downtrend -> short
                position = -1
                signals[i] = -0.25
    
    return signals