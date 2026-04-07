#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian(20) breakout with 1d EMA trend filter and volume confirmation
# Hypothesis: Breakouts from Donchian channels on 12h timeframe aligned with daily trend
# capture strong momentum moves. Volume confirmation reduces false breakouts.
# Works in both bull and bear markets by following the daily trend direction.
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.

name = "12h_donchian20_volume_ema_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    
    # Align 1d EMA to 12h
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Donchian(20) on 12h high/low
    def donchian_channels(high, low, window):
        upper = np.full_like(high, np.nan, dtype=float)
        lower = np.full_like(low, np.nan, dtype=float)
        for i in range(window-1, len(high)):
            upper[i] = np.max(high[i-window+1:i+1])
            lower[i] = np.min(low[i-window+1:i+1])
        return upper, lower
    
    upper, lower = donchian_channels(high, low, 20)
    
    # Volume average (20-period)
    def sma(data, window):
        result = np.full_like(data, np.nan, dtype=float)
        for i in range(window-1, len(data)):
            result[i] = np.mean(data[i-window+1:i+1])
        return result
    
    vol_avg = sma(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(ema_50_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_avg[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower or trend changes
            if close[i] < lower[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper or trend changes
            if close[i] > upper[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average
            vol_confirm = volume[i] > 1.5 * vol_avg[i]
            
            # Long breakout: price closes above Donchian upper in uptrend
            if close[i] > upper[i] and close[i] > ema_50_aligned[i] and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Short breakdown: price closes below Donchian lower in downtrend
            elif close[i] < lower[i] and close[i] < ema_50_aligned[i] and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals