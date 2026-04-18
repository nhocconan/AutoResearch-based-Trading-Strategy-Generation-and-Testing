#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrendVolume
Hypothesis: Trade 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation. 
Long when price breaks above Donchian upper (20-period high) with 1d EMA34 rising and volume > 1.5x 24-period average.
Short when price breaks below Donchian lower (20-period low) with 1d EMA34 falling and volume > 1.5x average.
Exit when price crosses opposite Donchian band or trend reverses.
Uses volatility-adjusted position sizing (0.25) to manage drawdown in volatile markets.
Works in bull by capturing breakouts, in bear by shorting breakdowns with trend filter.
Target: 20-40 trades/year via strict breakout + trend + volume confluence.
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA34
    ema_period = 34
    ema_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period-1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 / (ema_period + 1)) + (ema_1d[i-1] * (ema_period - 1) / (ema_period + 1))
    
    # Align 1d EMA34 to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Donchian channels on 4h
    donch_period = 20
    donch_high = np.full_like(high, np.nan)
    donch_low = np.full_like(low, np.nan)
    
    if len(high) >= donch_period:
        for i in range(donch_period-1, len(high)):
            donch_high[i] = np.max(high[i-donch_period+1:i+1])
            donch_low[i] = np.min(low[i-donch_period+1:i+1])
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donch_period, vol_period, ema_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]) or
            np.isnan(ema_1d_aligned[i-1])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend direction from 1d EMA34 slope
        ema_rising = ema_1d_aligned[i] > ema_1d_aligned[i-1]
        ema_falling = ema_1d_aligned[i] < ema_1d_aligned[i-1]
        
        if position == 0:
            # Long: break above Donchian high + rising trend + volume
            if close[i] > donch_high[i] and ema_rising and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low + falling trend + volume
            elif close[i] < donch_low[i] and ema_falling and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long: maintain position
            signals[i] = 0.25
            # Exit: price crosses below Donchian low OR trend turns down
            if close[i] < donch_low[i] or ema_falling:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short: maintain position
            signals[i] = -0.25
            # Exit: price crosses above Donchian high OR trend turns up
            if close[i] > donch_high[i] or ema_rising:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dTrendVolume"
timeframe = "4h"
leverage = 1.0