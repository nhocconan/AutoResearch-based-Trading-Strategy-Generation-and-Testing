#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: 12h chart strategy using daily Camarilla R1/S1 breakouts with 1d EMA trend filter and volume confirmation.
Works in bull/bear by following daily trend; volatility-based entries reduce false breakouts.
Target: 12-30 trades/year per symbol with strict entry conditions to minimize fee drag.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Calculate daily EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_34_1d[i-1]
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume SMA(20)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_sma[i]):
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
                    if close[i] > r1 and close[i] > ema_34_1d_aligned[i] and volume_confirm:
                        signals[i] = 0.25
                        position = 1
                    # Short: Break below S1 with downtrend and volume confirmation
                    elif close[i] < s1 and close[i] < ema_34_1d_aligned[i] and volume_confirm:
                        signals[i] = -0.25
                        position = -1
                elif position == 1:
                    # Exit: Close crosses back below EMA34_1d
                    if close[i] < ema_34_1d_aligned[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
                elif position == -1:
                    # Exit: Close crosses back above EMA34_1d
                    if close[i] > ema_34_1d_aligned[i]:
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