#!/usr/bin/env python3
"""
Hypothesis: 12-hour Camarilla Pivot Breakout with 1-day Trend Filter and Volume Confirmation.
Long when price breaks above R1 (Camarilla resistance) during 1-day uptrend with volume spike.
Short when price breaks below S1 (Camarilla support) during 1-day downtrend with volume spike.
Exit when price returns to previous day's close or trend reverses.
Designed for low trade frequency by requiring confluence of pivot breakout, trend alignment, and volume confirmation.
Works in both bull and bear markets by following the 1-day trend.
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
    
    # Calculate Camarilla pivot levels from previous 12h bar
    # Camarilla formulas: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_range = prev_high - prev_low
    r1 = prev_close + 1.1 * prev_range / 12
    s1 = prev_close - 1.1 * prev_range / 12
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_range[0] = np.nan
    r1[0] = np.nan
    s1[0] = np.nan
    
    # Load 1-day data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 34-period EMA on 1d close for trend (more stable than 20)
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average (higher threshold for fewer trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(prev_close[i]) or np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or 
            np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above R1 + 1-day uptrend + volume spike
            if close[i] > r1[i] and ema34_1d_aligned[i] > ema34_1d_aligned[i-1] and vol_spike:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S1 + 1-day downtrend + volume spike
            elif close[i] < s1[i] and ema34_1d_aligned[i] < ema34_1d_aligned[i-1] and vol_spike:
                signals[i] = -0.30
                position = -1
        else:
            # Exit: price returns to previous day's close or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price below previous close or 1d trend turns down
                if close[i] < prev_close[i] or ema34_1d_aligned[i] < ema34_1d_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price above previous close or 1d trend turns up
                if close[i] > prev_close[i] or ema34_1d_aligned[i] > ema34_1d_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "12H_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0