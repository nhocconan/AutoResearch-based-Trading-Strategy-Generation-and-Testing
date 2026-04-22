#!/usr/bin/env python3
"""
Hypothesis: 1-hour timeframe with 4-hour and 1-day trend filters for high-probability entries.
Long when price breaks above 4h Donchian upper band with volume spike and 1-day EMA50 uptrend.
Short when price breaks below 4h Donchian lower band with volume spike and 1-day EMA50 downtrend.
Exit when price returns to 4h Donchian middle band or trend reverses.
Uses 4h structure for direction, 1h for entry timing, and volume for confirmation to reduce false breaks.
Designed for low trade frequency (15-35/year) by requiring multiple confirmations.
Works in bull markets via trend following and in bear markets via short signals during downtrends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4-hour data for Donchian channels - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 4-hour Donchian channels (20-period)
    high_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    dc_upper_4h = high_20_4h
    dc_lower_4h = low_20_4h
    dc_middle_4h = (dc_upper_4h + dc_lower_4h) / 2.0
    
    # Align 4h Donchian levels to 1h timeframe
    dc_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, dc_upper_4h)
    dc_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, dc_lower_4h)
    dc_middle_4h_aligned = align_htf_to_ltf(prices, df_4h, dc_middle_4h)
    
    # Load 1-day data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike detection (1h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x 20-period average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(dc_upper_4h_aligned[i]) or np.isnan(dc_lower_4h_aligned[i]) or 
            np.isnan(dc_middle_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 4h Donchian upper band with volume spike and 1-day EMA50 uptrend
            if (close[i] > dc_upper_4h_aligned[i] and 
                vol_spike[i] and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1]):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below 4h Donchian lower band with volume spike and 1-day EMA50 downtrend
            elif (close[i] < dc_lower_4h_aligned[i] and 
                  vol_spike[i] and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1]):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to 4h Donchian middle band OR 1-day EMA50 turns down
                if (close[i] <= dc_middle_4h_aligned[i] or 
                    ema50_1d_aligned[i] < ema50_1d_aligned[i-1]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to 4h Donchian middle band OR 1-day EMA50 turns up
                if (close[i] >= dc_middle_4h_aligned[i] or 
                    ema50_1d_aligned[i] > ema50_1d_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_DonchianBreakout_4hDir_1dTrend_Volume"
timeframe = "1h"
leverage = 1.0