#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 Breakout + 1d EMA34 Trend Filter + Volume Spike
- Uses 12h Camarilla pivot levels (R1, S1) from prior 1d session for breakout entries
- 1d EMA34 defines medium-term trend filter: only trade in direction of daily trend
- Volume confirmation (> 2.0x 24-period average) filters weak signals
- Stoploss via signal=0 when price closes below/above 1d EMA34
- Designed for 12h timeframe targeting 12-25 trades/year (50-100 over 4 years)
- Works in both bull and bear markets by combining structure (pivots) with trend/volume filters
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h Camarilla levels from prior 1d session
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    df_1d_prev_close = df_1d['close'].shift(1).values
    df_1d_prev_high = df_1d['high'].shift(1).values
    df_1d_prev_low = df_1d['low'].shift(1).values
    
    camarilla_r1 = df_1d_prev_close + 1.1 * (df_1d_prev_high - df_1d_prev_low) / 12
    camarilla_s1 = df_1d_prev_close - 1.1 * (df_1d_prev_high - df_1d_prev_low) / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: > 2.0x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 24)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close breaks above Camarilla R1 AND above 1d EMA34 AND volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Camarilla S1 AND below 1d EMA34 AND volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price closes below/above 1d EMA34
            exit_signal = False
            
            if position == 1:
                # Exit long when price closes below 1d EMA34
                if close[i] < ema_34_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price closes above 1d EMA34
                if close[i] > ema_34_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0