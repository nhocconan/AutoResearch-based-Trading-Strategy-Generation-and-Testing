#!/usr/bin/env python3
# 1h_Camarilla_R2_S2_Breakout_4hTrend_1dVolume
# Hypothesis: 1h timeframe strategy using 4h Camarilla R2/S2 breakouts with 1d EMA50 trend filter and volume spike confirmation.
# Uses higher timeframe (4h) for signal direction (trend and key levels) and 1h for precise entry timing.
# In bull markets: buys breakouts above R2 in uptrend (price > EMA50) with volume confirmation.
# In bear markets: sells breakdowns below S2 in downtrend (price < EMA50) with volume confirmation.
# Volume filter ensures only strong breakouts are traded, reducing false signals and trade frequency.
# Target: 15-35 trades/year per symbol to avoid fee drag.

name = "1h_Camarilla_R2_S2_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

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
    
    # Get 4h data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Previous day's values for Camarilla calculation (using 4h data as proxy for daily)
    ph = np.concatenate([[high_4h[0]], high_4h[:-1]])  # previous high
    pl = np.concatenate([[low_4h[0]], low_4h[:-1]])   # previous low
    pc = np.concatenate([[close_4h[0]], close_4h[:-1]]) # previous close
    
    # Calculate Camarilla levels (R2, S2 are the key breakout levels)
    rang = ph - pl
    r2 = pc + 1.1 * rang * 1.0833  # R2 = Close + 1.1 * (High-Low) * 1.0833
    s2 = pc - 1.1 * rang * 1.0833  # S2 = Close - 1.1 * (High-Low) * 1.0833
    
    # Align Camarilla levels to 1h timeframe
    r2_aligned = align_htf_to_ltf(prices, df_4h, r2)
    s2_aligned = align_htf_to_ltf(prices, df_4h, s2)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (ema_50_1d[i-1] * 49 + close_1d[i]) / 50
    
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    # Session filter: 08-20 UTC (inclusive)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure volume MA and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ratio[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R2 AND uptrend (price > EMA50) AND volume spike
            if (close[i] > r2_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below S2 AND downtrend (price < EMA50) AND volume spike
            elif (close[i] < s2_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S2 OR trend reversal (price < EMA50)
            if close[i] < s2_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price breaks above R2 OR trend reversal (price > EMA50)
            if close[i] > r2_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals