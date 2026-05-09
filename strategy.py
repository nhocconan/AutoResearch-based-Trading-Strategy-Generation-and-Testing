# 1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeS
# Hypothesis: 1h breakouts from daily Camarilla R1/S1 levels with 4h EMA50 trend filter and volume spike confirmation.
# Uses daily price structure for direction, 4h trend filter to avoid counter-trend trades, and 1h for precise entry timing.
# Session filter (08-20 UTC) reduces noise. Target: 15-37 trades/year per symbol to minimize fee drag.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeS"
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
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for Camarilla calculation
    ph = np.concatenate([[high_1d[0]], high_1d[:-1]])
    pl = np.concatenate([[low_1d[0]], low_1d[:-1]])
    pc = np.concatenate([[close_1d[0]], close_1d[:-1]])
    
    # Calculate daily Camarilla levels (R1, S1)
    rang = ph - pl
    r1 = pc + 1.1 * rang * 1.0833
    s1 = pc - 1.1 * rang * 1.0833
    
    # Align daily Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 50:
        ema_50_4h[49] = np.mean(close_4h[0:50])
        for i in range(50, len(close_4h)):
            ema_50_4h[i] = (ema_50_4h[i-1] * 49 + close_4h[i]) / 50
    
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume spike filter: current 1h volume / 24-period average (24h)
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 24:
        vol_ma[23] = np.mean(volume[0:24])
        for i in range(24, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 23 + volume[i]) / 24
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = max(24, 50)  # Ensure volume MA and EMA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_ratio[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Enter long: price breaks above R1 AND uptrend (price > 4h EMA50) AND volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.20
                position = 1
                bars_since_entry = 0
            # Enter short: price breaks below S1 AND downtrend (price < 4h EMA50) AND volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.20
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Minimum holding period: 3 bars
            if bars_since_entry < 3:
                signals[i] = 0.20
            else:
                # Exit long: price breaks below S1 OR trend reversal (price < 4h EMA50)
                if close[i] < s1_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.20
        
        elif position == -1:
            # Minimum holding period: 3 bars
            if bars_since_entry < 3:
                signals[i] = -0.20
            else:
                # Exit short: price breaks above R1 OR trend reversal (price > 4h EMA50)
                if close[i] > r1_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.20
    
    return signals