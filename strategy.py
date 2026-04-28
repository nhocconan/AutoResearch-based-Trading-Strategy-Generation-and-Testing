#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS_v2
Hypothesis: Focus on high-probability breakouts at daily Camarilla R1/S1 levels with 12h EMA50 trend filter and volume confirmation on 4h timeframe.
Reduced trade frequency by requiring volume surge to be >2.5x average (vs 2.0x) and adding a minimum hold period of 4 bars to avoid whipsaw.
Targets 20-50 trades/year by requiring multiple confluence factors (breakout, trend, volume) to reduce false signals and work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla R1 and S1 levels
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all higher timeframe data to 4h
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Trend filter: price > EMA50 = bullish, < EMA50 = bearish
    h12_uptrend = close > ema_50_12h_aligned
    h12_downtrend = close < ema_50_12h_aligned
    
    # Volume confirmation: current volume > 2.5x 20-period average (increased threshold)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 2.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = 200  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        # Entry conditions with trend alignment and volume surge
        # Long: price breaks above R1 + 12h uptrend + volume surge
        long_entry = (close[i] > R1_aligned[i] and 
                     h12_uptrend[i] and 
                     volume_surge[i])
        
        # Short: price breaks below S1 + 12h downtrend + volume surge
        short_entry = (close[i] < S1_aligned[i] and 
                      h12_downtrend[i] and 
                      volume_surge[i])
        
        # Exit on opposite level break with volume surge
        long_exit = close[i] < S1_aligned[i] and volume_surge[i]
        short_exit = close[i] > R1_aligned[i] and volume_surge[i]
        
        if long_entry and position <= 0 and bars_since_entry >= 4:
            signals[i] = 0.25
            position = 1
            bars_since_entry = 0
        elif short_entry and position >= 0 and bars_since_entry >= 4:
            signals[i] = -0.25
            position = -1
            bars_since_entry = 0
        elif long_exit and position == 1 and bars_since_entry >= 4:
            signals[i] = -0.25  # Reverse to short
            position = -1
            bars_since_entry = 0
        elif short_exit and position == -1 and bars_since_entry >= 4:
            signals[i] = 0.25   # Reverse to long
            position = 1
            bars_since_entry = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS_v2"
timeframe = "4h"
leverage = 1.0