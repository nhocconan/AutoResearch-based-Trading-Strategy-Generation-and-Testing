#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 1d Camarilla R1/S1 breakout + volume confirmation + session filter
# Uses daily pivot structure for direction, 1h for entry timing. Designed to work in both bull/bear
# by capturing intraday momentum within established daily support/resistance levels.
# Target: 15-35 trades/year to avoid fee drag.

name = "1h_1d_Camarilla_R1S1_Breakout_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    prev_close = np.roll(close_1d, 1)
    prev_close[0] = np.nan
    prev_high = np.roll(high_1d, 1)
    prev_high[0] = np.nan
    prev_low = np.roll(low_1d, 1)
    prev_low[0] = np.nan
    
    # Pivot = (H + L + C) / 3
    pivot = (prev_high + prev_low + prev_close) / 3.0
    # R1 = C + (H - L) * 1.1 / 12
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 12.0
    # S1 = C - (H - L) * 1.1 / 12
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 12.0
    
    # Align to 1h timeframe
    r1_1h = align_htf_to_ltf(prices, df_1d, r1)
    s1_1h = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: current volume > 2.0x 20-period average (stricter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Time filter: 08-20 UTC (reduces noise outside active sessions)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    time_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if not time_filter[i]:
            signals[i] = 0.0
            continue
            
        if np.isnan(r1_1h[i]) or np.isnan(s1_1h[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        volume_confirmed = vol > 2.0 * vol_ma  # Stricter volume filter
        
        if position == 0:
            # Long: Price breaks above R1 with volume
            if price > r1_1h[i] and volume_confirmed:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below S1 with volume
            elif price < s1_1h[i] and volume_confirmed:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: Price returns below S1 (reversal signal)
            if price < s1_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: Price returns above R1 (reversal signal)
            if price > r1_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals