#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout with 1d Volume Confirmation
# Hypothesis: Donchian(20) breakouts with 1d volume spike and 1d close > 50 EMA filter capture strong trends.
# Uses 1d volume > 1.5x 20-period average for confirmation and avoids counter-trend trades.
# Target: 20-30 trades/year (80-120 total over 4 years) to minimize fee drag.

name = "4h_donchian_breakout_1d_volume_v2"
timeframe = "4h"
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
    
    # Get 1d data for volume and EMA filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) on close
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    # Donchian(20) on 4h high/low
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(40, n):
        # Skip if required data not available
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_20_aligned[i]) or \
           np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band or 1d EMA turns down
            if close[i] <= lowest_low[i] or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band or 1d EMA turns up
            if close[i] >= highest_high[i] or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Volume confirmation: 1d volume > 1.5x 20-day average
            vol_confirm = vol_1d[i] > 1.5 * vol_avg_20_aligned[i] if not np.isnan(vol_1d[i]) else False
            
            # Long: price breaks above Donchian upper band with volume and above 1d EMA
            if close[i] > highest_high[i] and vol_confirm and close[i] > ema_50_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian lower band with volume and below 1d EMA
            elif close[i] < lowest_low[i] and vol_confirm and close[i] < ema_50_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals