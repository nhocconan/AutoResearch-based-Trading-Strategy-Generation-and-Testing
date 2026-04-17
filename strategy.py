#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_R1_S1_Breakout_With_Volume_and_Trend_Filter_v1
Based on Camarilla pivot levels calculated from weekly OHLC. 
Long when price breaks above R1 with volume spike and weekly close above weekly open (bullish weekly candle).
Short when price breaks below S1 with volume spike and weekly close below weekly open (bearish weekly candle).
Exit when price returns to the weekly close level (pivot point equivalent).
Uses volume spike (volume > 1.5 * 20-day average volume) for confirmation.
Designed to capture meaningful breakouts with institutional volume backing.
Target: 30-100 total trades over 4 years (7-25/year).
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
    
    # === Weekly OHLC for Camarilla calculation ===
    df_1w = get_htf_data(prices, '1w')
    weekly_open = df_1w['open'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate Camarilla levels for each day based on prior week
    # R1 = weekly_close + (weekly_high - weekly_low) * 1.1 / 12
    # S1 = weekly_close - (weekly_high - weekly_low) * 1.1 / 12
    weekly_range = weekly_high - weekly_low
    camarilla_r1 = weekly_close + weekly_range * 1.1 / 12
    camarilla_s1 = weekly_close - weekly_range * 1.1 / 12
    
    # Align weekly levels to daily timeframe (using prior week's values)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close)
    weekly_open_aligned = align_htf_to_ltf(prices, df_1w, weekly_open)
    
    # === Volume spike filter: volume > 1.5 * 20-day average volume ===
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(weekly_close_aligned[i]) or 
            np.isnan(weekly_open_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R1, volume spike, weekly close above weekly open (bullish weekly candle)
            if (close[i] > camarilla_r1_aligned[i] and 
                volume_spike[i] and 
                weekly_close_aligned[i] > weekly_open_aligned[i]):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below S1, volume spike, weekly close below weekly open (bearish weekly candle)
            elif (close[i] < camarilla_s1_aligned[i] and 
                  volume_spike[i] and 
                  weekly_close_aligned[i] < weekly_open_aligned[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: return to weekly close level (pivot equivalent)
        elif position == 1:
            # Exit long: price crosses below weekly close
            if close[i] < weekly_close_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above weekly close
            if close[i] > weekly_close_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_Pivot_R1_S1_Breakout_With_Volume_and_Trend_Filter_v1"
timeframe = "1d"
leverage = 1.0