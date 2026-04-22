#!/usr/bin/env python3

"""
Hypothesis: 4-hour Camarilla Pivot Point R1/S1 breakout with 12-hour EMA50 trend filter and volume confirmation.
Only trade breakouts in the direction of the 12-hour EMA50 trend when price breaks above R1 (long) or below S1 (short)
with volume > 2x 20-period average. Camarilla levels derived from prior 12-hour bar's high-low range.
Designed for low trade frequency (20-40 trades/year) by requiring trend alignment, pivot breakout, and volume spike.
Works in both bull and bear markets by following the 12-hour EMA50 trend direction, which adapts to market conditions.
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
    
    # 12-hour EMA50 for trend direction
    close_s = pd.Series(close)
    ema50_12h = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Load 12-hour data for Camarilla pivots - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 12h bar
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Typical price for pivot calculation
    typical_price = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    
    # Camarilla levels: R1 = close + (range * 1.1/12), S1 = close - (range * 1.1/12)
    r1 = close_12h + (range_12h * 1.1 / 12)
    s1 = close_12h - (range_12h * 1.1 / 12)
    
    # Align to 4-hour timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: EMA50 uptrend + price breaks above R1 + volume spike
            if ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and close[i] > r1_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: EMA50 downtrend + price breaks below S1 + volume spike
            elif ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and close[i] < s1_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: EMA50 trend reversal or price returns to prior close level
            exit_signal = False
            
            if position == 1:
                # Exit long: EMA50 turns down or price closes below prior 12h close
                if ema50_12h_aligned[i] < ema50_12h_aligned[i-1] or close[i] < close_12h[-1] if len(close_12h) > 0 else False:
                    exit_signal = True
            else:  # position == -1
                # Exit short: EMA50 turns up or price closes above prior 12h close
                if ema50_12h_aligned[i] > ema50_12h_aligned[i-1] or close[i] > close_12h[-1] if len(close_12h) > 0 else False:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0