#!/usr/bin/env python3
"""
Hypothesis: 4-hour Camarilla R1/S1 Breakout with 1-day EMA34 Trend and Volume Spike.
Long when price breaks above R1 with volume spike and 1d EMA34 rising.
Short when price breaks below S1 with volume spike and 1d EMA34 falling.
Exit when price crosses back below EMA34 or R1/S1 levels.
Designed for low trade frequency by requiring confluence of price level, trend, and volume.
Works in both bull and bear markets by following the daily trend.
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
    
    # Calculate Camarilla levels from previous day
    # Use daily OHLC from previous completed day
    # For simplicity, we'll use current bar's high/low/close as approximation
    # In practice, Camarilla uses previous day's range
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # handle first bar
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    # Camarilla levels
    range_prev = prev_high - prev_low
    R1 = prev_close + range_prev * 1.1 / 12
    S1 = prev_close - range_prev * 1.1 / 12
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 34-period EMA on 1d close for trend
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(R1[i]) or np.isnan(S1[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above R1, volume spike, 1d EMA34 rising
            if (close[i] > R1[i] and vol_spike and 
                ema34_1d_aligned[i] > ema34_1d_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, volume spike, 1d EMA34 falling
            elif (close[i] < S1[i] and vol_spike and 
                  ema34_1d_aligned[i] < ema34_1d_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses back below EMA34 or breaks opposite level
            exit_signal = False
            
            if position == 1:
                # Exit long: price < EMA34 or breaks below S1
                if close[i] < ema34_1d_aligned[i] or close[i] < S1[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price > EMA34 or breaks above R1
                if close[i] > ema34_1d_aligned[i] or close[i] > R1[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R1_S1_Breakout_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0