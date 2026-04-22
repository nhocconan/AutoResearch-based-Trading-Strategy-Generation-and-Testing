#!/usr/bin/env python3
"""
Hypothesis: 4-hour Camarilla R1/S1 Breakout with 1-day Volume Spike and 1-day EMA34 Trend.
Long when price breaks above R1 with volume spike and EMA34 uptrend.
Short when price breaks below S1 with volume spike and EMA34 downtrend.
Exit when price crosses H3/L3 or trend reverses.
Camarilla levels provide precise support/resistance; volume spike confirms institutional interest.
Works in both bull and bear by following institutional volume with trend filter.
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
    
    # Load 1-day data for Camarilla, EMA34, and volume - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's range
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Previous day's values (shift by 1 to avoid look-ahead)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = np.nan  # First day has no previous
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Camarilla levels
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    H3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    L3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 4h
    R1_4h = align_htf_to_ltf(prices, df_1d, R1)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1)
    H3_4h = align_htf_to_ltf(prices, df_1d, H3)
    L3_4h = align_htf_to_ltf(prices, df_1d, L3)
    
    # 1-day EMA34 trend
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # 1-day volume spike (current vs 20-period average)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > vol_ma_20 * 1.5  # 50% above average
    vol_spike_4h = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA34 warmup
        # Skip if any data not ready
        if (np.isnan(R1_4h[i]) or np.isnan(S1_4h[i]) or 
            np.isnan(H3_4h[i]) or np.isnan(L3_4h[i]) or
            np.isnan(ema_34_4h[i]) or np.isnan(vol_spike_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1, volume spike, EMA34 uptrend
            if (close[i] > R1_4h[i] and 
                vol_spike_4h[i] and 
                close[i] > ema_34_4h[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1, volume spike, EMA34 downtrend
            elif (close[i] < S1_4h[i] and 
                  vol_spike_4h[i] and 
                  close[i] < ema_34_4h[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below H3 or trend turns down
                if (close[i] < H3_4h[i] or 
                    close[i] < ema_34_4h[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above L3 or trend turns up
                if (close[i] > L3_4h[i] or 
                    close[i] > ema_34_4h[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0