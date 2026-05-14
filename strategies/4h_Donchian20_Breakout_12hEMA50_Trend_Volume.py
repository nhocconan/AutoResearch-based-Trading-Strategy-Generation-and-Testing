#!/usr/bin/env python3
# 4h_Donchian20_Breakout_12hEMA50_Trend_Volume
# Hypothesis: Donchian(20) breakout on 4h with 12h EMA50 trend filter and volume confirmation.
# Long when 12h trend up and price breaks above 20-period high with volume > 1.5x average.
# Short when 12h trend down and price breaks below 20-period low with volume > 1.5x average.
# Trend filter reduces whipsaw in ranging markets, works in both bull and bear cycles.
# Target: 20-50 trades/year per symbol with disciplined risk management.

name = "4h_Donchian20_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 50:
        ema50_12h[49] = np.mean(close_12h[0:50])
        for i in range(50, len(close_12h)):
            ema50_12h[i] = (close_12h[i] * 2 + ema50_12h[i-1] * 48) / 50
    
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Donchian channels (20-period high/low)
    donchian_high = np.full_like(close, np.nan)
    donchian_low = np.full_like(close, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume filter: current volume vs 20-period average
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
    
    start_idx = max(50, 20, 20)  # Need 12h EMA, Donchian, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 12h trend
        trend_up = close[i] > ema50_12h_aligned[i]
        
        if position == 0:
            # Enter long: 12h trend up + price breaks above Donchian high + volume confirmation
            if trend_up and close[i] > donchian_high[i] and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: 12h trend down + price breaks below Donchian low + volume confirmation
            elif not trend_up and close[i] < donchian_low[i] and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: 12h trend turns down or price breaks below Donchian low
            if not trend_up or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: 12h trend turns up or price breaks above Donchian high
            if trend_up or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals