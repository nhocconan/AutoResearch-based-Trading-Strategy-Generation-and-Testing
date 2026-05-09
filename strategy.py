#!/usr/bin/env python3
# 4h_Donchian_Breakout_Volume_Trend
# Hypothesis: Combines 4h Donchian channel breakout with 1d EMA trend filter and volume confirmation.
# Long when price breaks above 20-period upper Donchian band, price > 1d EMA50, and volume > 2x average.
# Short when price breaks below 20-period lower Donchian band, price < 1d EMA50, and volume > 2x average.
# Exits on opposite Donchian band break or volume drop below average.
# Designed to capture strong trends with volume confirmation, reducing false breakouts.
# Target: 20-35 trades/year per symbol with disciplined risk to avoid overtrading.

name = "4h_Donchian_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    ema_50 = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50[i] = (close_1d[i] * 0.0377) + (ema_50[i-1] * 0.9623)
    
    # Align daily EMA to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 4h Donchian channels (20-period)
    upper_donchian = np.full_like(high, np.nan)
    lower_donchian = np.full_like(low, np.nan)
    
    if len(high) >= 20 and len(low) >= 20:
        for i in range(19, len(high)):
            upper_donchian[i] = np.max(high[i-19:i+1])
            lower_donchian[i] = np.min(low[i-19:i+1])
    
    # Volume filter: 4h volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or \
           np.isnan(ema_50_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price breaks above upper Donchian + above 1d EMA50 + volume confirmation
            if close[i] > upper_donchian[i] and close[i] > ema_50_aligned[i] and volume_ratio[i] > 2.0:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below lower Donchian + below 1d EMA50 + volume confirmation
            elif close[i] < lower_donchian[i] and close[i] < ema_50_aligned[i] and volume_ratio[i] > 2.0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price breaks below lower Donchian OR volume drops below average
            if close[i] < lower_donchian[i] or volume_ratio[i] < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price breaks above upper Donchian OR volume drops below average
            if close[i] > upper_donchian[i] or volume_ratio[i] < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals