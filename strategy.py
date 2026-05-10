#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS
Hypothesis: Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike confirmation.
Works in bull/bear by following 12h trend; volatility-based entries reduce false breakouts.
Target: 20-30 trades/year per symbol with strict entry conditions to minimize fee drag.
"""

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema_50_12h[49] = np.mean(close_12h[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_12h)):
            ema_50_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema_50_12h[i-1]
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate volume SMA(20)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirm = volume[i] > 2.0 * vol_sma[i]
        
        # Calculate Camarilla levels from previous day's OHLC
        if i > 0:
            prev_day_idx = i - 1
            # Check if previous bar is from previous day
            curr_date = pd.Timestamp(prices['open_time'].iloc[i]).date()
            prev_date = pd.Timestamp(prices['open_time'].iloc[prev_day_idx]).date()
            
            if curr_date != prev_date:
                # Previous bar is from previous day, use its OHLC
                ph = prices['high'].iloc[prev_day_idx]
                pl = prices['low'].iloc[prev_day_idx]
                pc = prices['close'].iloc[prev_day_idx]
                
                # Camarilla levels
                range_ = ph - pl
                r1 = pc + (range_ * 1.1 / 12)
                s1 = pc - (range_ * 1.1 / 12)
                
                if position == 0:
                    # Long: Break above R1 with uptrend and volume confirmation
                    if close[i] > r1 and close[i] > ema_50_12h_aligned[i] and volume_confirm:
                        signals[i] = 0.25
                        position = 1
                    # Short: Break below S1 with downtrend and volume confirmation
                    elif close[i] < s1 and close[i] < ema_50_12h_aligned[i] and volume_confirm:
                        signals[i] = -0.25
                        position = -1
                elif position == 1:
                    # Exit: Close crosses back below EMA50_12h
                    if close[i] < ema_50_12h_aligned[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
                elif position == -1:
                    # Exit: Close crosses back above EMA50_12h
                    if close[i] > ema_50_12h_aligned[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
            else:
                # Same day, hold current position
                if position == 1:
                    signals[i] = 0.25
                elif position == -1:
                    signals[i] = -0.25
                else:
                    signals[i] = 0.0
        else:
            # First bar, hold flat
            signals[i] = 0.0
    
    return signals