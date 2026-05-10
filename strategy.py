#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_Volume
Hypothesis: Weekly EMA200 trend filter combined with daily Camarilla R1/S1 breakouts and volume confirmation on 12h timeframe.
Uses higher timeframe trend to avoid counter-trend trades, works in both bull and bear markets by following weekly trend.
Target: 15-25 trades/year per symbol with strict entry conditions to minimize fee drag.
"""

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
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
    
    # Calculate 1w EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_200_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 200:
        ema_200_1w[199] = np.mean(close_1w[:200])
        alpha = 2 / (200 + 1)
        for i in range(200, len(close_1w)):
            ema_200_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_200_1w[i-1]
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate daily volume SMA(20)
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_sma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_sma_1d[i] = np.mean(volume_1d[i-20:i])
    vol_sma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 200)
    
    for i in range(start_idx, n):
        if np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_sma_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average daily volume
        # Need to map 12h bar to daily volume (simplified: use current day's average volume)
        vol_confirm = volume[i] > 1.5 * vol_sma_1d_aligned[i]
        
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
                    # Long: Break above R1 with uptrend (price > weekly EMA200) and volume confirmation
                    if close[i] > r1 and close[i] > ema_200_1w_aligned[i] and vol_confirm:
                        signals[i] = 0.25
                        position = 1
                    # Short: Break below S1 with downtrend (price < weekly EMA200) and volume confirmation
                    elif close[i] < s1 and close[i] < ema_200_1w_aligned[i] and vol_confirm:
                        signals[i] = -0.25
                        position = -1
                elif position == 1:
                    # Exit: Close crosses back below weekly EMA200
                    if close[i] < ema_200_1w_aligned[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
                elif position == -1:
                    # Exit: Close crosses back above weekly EMA200
                    if close[i] > ema_200_1w_aligned[i]:
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