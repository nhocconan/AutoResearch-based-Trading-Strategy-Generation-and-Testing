#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeControl
Hypothesis: 12h breakouts at daily Camarilla R1/S1 levels with 1d trend filter (EMA34) and volume control.
Targets 12-37 trades/year (~50-150 total) by requiring breakout + trend + volume confirmation.
Works in bull/bear: buys strength in uptrends, sells weakness in downtrends with volatility filter.
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
    
    # Get 1d data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    R1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    S1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume control: avoid low-volume chop
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    vol_ratio = volume / np.maximum(vol_ma_50, 1e-8)
    
    # Align HTF data to 12h
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Warmup
    
    for i in range(start_idx, n):
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Long: break above R1 + 1d uptrend + volume > 0.5x average (avoid dead cat bounces)
        long_entry = (close[i] > R1_aligned[i] and 
                     close[i] > ema_34_1d_aligned[i] and 
                     vol_ratio[i] > 0.5)
        
        # Short: break below S1 + 1d downtrend + volume > 0.5x average
        short_entry = (close[i] < S1_aligned[i] and 
                      close[i] < ema_34_1d_aligned[i] and 
                      vol_ratio[i] > 0.5)
        
        # Exit: reverse on opposite level break with volume confirmation
        long_exit = close[i] < S1_aligned[i] and vol_ratio[i] > 0.5
        short_exit = close[i] > R1_aligned[i] and vol_ratio[i] > 0.5
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeControl"
timeframe = "12h"
leverage = 1.0