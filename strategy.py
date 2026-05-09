#!/usr/bin/env python3
# 12h_Donchian20_Breakout_1dTrend_VolumeFilter
# Hypothesis: Combines 4-hour Donchian breakouts with 1-day trend filter and volume confirmation.
# Uses Donchian channel breakouts (20-period high/low) on the 12h chart, with trend direction
# determined by 1-day EMA50, and volume spike for confirmation. Designed to capture medium-term
# trends while avoiding false breakouts in low-volume or ranging conditions.
# Target: 15-25 trades/year per symbol with disciplined risk management.

name = "12h_Donchian20_Breakout_1dTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1-day EMA50 for trend filter
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 + ema_50_1d[i-1] * 49) / 51
    
    # Align 1-day EMA50 to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = np.full_like(high, np.nan)
    donchian_low = np.full_like(low, np.nan)
    
    if len(high) >= 20:
        donchian_high[19] = np.max(high[0:20])
        donchian_low[19] = np.min(low[0:20])
        for i in range(20, len(high)):
            donchian_high[i] = max(donchian_high[i-1], high[i])
            donchian_low[i] = min(donchian_low[i-1], low[i])
            # Remove the oldest value from the window
            if i >= 40:
                donchian_high[i] = max(high[i-19:i+1])
                donchian_low[i] = min(low[i-19:i+1])
    
    # Volume filter: 12h volume / 20-period average volume
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
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1-day EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Enter long: Price breaks above Donchian high AND uptrend AND volume confirmation
            if close[i] > donchian_high[i] and uptrend and volume_ratio[i] > 1.8:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below Donchian low AND downtrend AND volume confirmation
            elif close[i] < donchian_low[i] and downtrend and volume_ratio[i] > 1.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price breaks below Donchian low OR trend reversal
            if close[i] < donchian_low[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price breaks above Donchian high OR trend reversal
            if close[i] > donchian_high[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals