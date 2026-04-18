#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrendVolume
Hypothesis: Buy breakouts above 20-period Donchian high on 4h when 1d EMA50 is rising (uptrend) and volume > 1.5x 20-period average. Sell/short breakdowns below 20-period Donchian low when 1d EMA50 is falling (downtrend) and volume > 1.5x average. Uses 1d trend filter to avoid counter-trend trades and volume confirmation to avoid false breakouts. Designed for 15-25 trades/year on 4f to minimize fee drag. Works in bull markets via breakouts and in bear via breakdowns following the higher timeframe trend.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema_period = 50
    ema_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period-1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 / (ema_period + 1)) + (ema_1d[i-1] * (ema_period - 1) / (ema_period + 1))
    
    # Align 1d EMA50 to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Donchian channels on 4h (20-period)
    donchian_period = 20
    donchian_high = np.full_like(high, np.nan)
    donchian_low = np.full_like(low, np.nan)
    
    if len(high) >= donchian_period:
        for i in range(donchian_period-1, len(high)):
            donchian_high[i] = np.max(high[i-donchian_period+1:i+1])
            donchian_low[i] = np.min(low[i-donchian_period+1:i+1])
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_period = 20
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_period, vol_period, ema_period)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: EMA50 rising/falling
        if i > 0 and not np.isnan(ema_1d_aligned[i-1]):
            ema_rising = ema_1d_aligned[i] > ema_1d_aligned[i-1]
            ema_falling = ema_1d_aligned[i] < ema_1d_aligned[i-1]
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long: breakout above Donchian high + uptrend + volume
            if close[i] > donchian_high[i] and ema_rising and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian low + downtrend + volume
            elif close[i] < donchian_low[i] and ema_falling and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: breakdown below Donchian low or trend turns down
            if close[i] < donchian_low[i] or (i > 0 and not np.isnan(ema_1d_aligned[i-1]) and ema_1d_aligned[i] < ema_1d_aligned[i-1]):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: breakout above Donchian high or trend turns up
            if close[i] > donchian_high[i] or (i > 0 and not np.isnan(ema_1d_aligned[i-1]) and ema_1d_aligned[i] > ema_1d_aligned[i-1]):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dTrendVolume"
timeframe = "4h"
leverage = 1.0