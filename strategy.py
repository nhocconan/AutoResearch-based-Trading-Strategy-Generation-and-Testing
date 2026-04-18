#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA34 trend filter and volume confirmation.
Uses price channel breakouts (proven on SOL) combined with higher-timeframe trend filter
to reduce false signals and whipsaws. Designed for 20-40 trades/year to minimize fee drag.
Works in bull markets via breakouts and bear markets via trend-filtered short signals.
"""

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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate EMA(34) on 12h close
    ema_34_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 34:
        ema_34_12h[33] = np.mean(close_12h[:34])
        for i in range(34, len(close_12h)):
            ema_34_12h[i] = (close_12h[i] * 2/35) + (ema_34_12h[i-1] * 33/35)
    
    # Align 12h EMA to 4h timeframe
    ema_34_12h_4h = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 4h ATR(14)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[:14])
        else:
            atr[i] = (tr[i] * 1/14) + (atr[i-1] * 13/14)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Calculate Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # need EMA, ATR, volume MA, Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_12h_4h[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above/below 12h EMA34
        trend_up = close[i] > ema_34_12h_4h[i]
        trend_down = close[i] < ema_34_12h_4h[i]
        
        if position == 0:
            # Long entry: close above Donchian high + 0.1*ATR, with volume and trend filter
            if (close[i] > donchian_high[i] + 0.1 * atr[i] and 
                vol_confirmed and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: close below Donchian low - 0.1*ATR, with volume and trend filter
            elif (close[i] < donchian_low[i] - 0.1 * atr[i] and 
                  vol_confirmed and 
                  trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: close below Donchian low or ATR-based stop
            if close[i] < donchian_low[i] - 0.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above Donchian high or ATR-based stop
            if close[i] > donchian_high[i] + 0.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA34_VolumeFilter"
timeframe = "4h"
leverage = 1.0