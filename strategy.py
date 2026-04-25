#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation. 
Long when price breaks above R1 with 1d EMA34 uptrend and volume > 1.5x average. 
Short when price breaks below S1 with 1d EMA34 downtrend and volume > 1.5x average.
Exit on opposite Camarilla level touch (R3/S3) or close back inside R1/S1.
Uses discrete sizing (0.25) to minimize fees. Target: 20-40 trades/year.
Works in bull markets via breakouts and in bear markets via mean reversion at extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate previous day's Camarilla levels (using 1d OHLC)
    # Camarilla equations:
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # R2 = close + ((high - low) * 1.1/6)
    # R1 = close + ((high - low) * 1.1/12)
    # S1 = close - ((high - low) * 1.1/12)
    # S2 = close - ((high - low) * 1.1/6)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    
    # Handle first bar
    prev_close_1d[0] = close_1d[0]
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    R1_1d = prev_close_1d + ((prev_high_1d - prev_low_1d) * 1.1 / 12)
    S1_1d = prev_close_1d - ((prev_high_1d - prev_low_1d) * 1.1 / 12)
    R3_1d = prev_close_1d + ((prev_high_1d - prev_low_1d) * 1.1 / 4)
    S3_1d = prev_close_1d - ((prev_high_1d - prev_low_1d) * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above R1 + 1d EMA34 uptrend + volume spike
            long_signal = (close[i] > R1_aligned[i]) and (ema_34_aligned[i] > ema_34_aligned[i-1]) and volume_spike[i]
            # Short: price breaks below S1 + 1d EMA34 downtrend + volume spike
            short_signal = (close[i] < S1_aligned[i]) and (ema_34_aligned[i] < ema_34_aligned[i-1]) and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Price touches R3 (take profit)
            # 2. Price closes back inside R1/S1 range (mean reversion)
            exit_signal = (close[i] >= R3_aligned[i]) or (close[i] <= S1_aligned[i] and close[i] >= R1_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Price touches S3 (take profit)
            # 2. Price closes back inside R1/S1 range (mean reversion)
            exit_signal = (close[i] <= S3_aligned[i]) or (close[i] >= S1_aligned[i] and close[i] <= R1_aligned[i])
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0