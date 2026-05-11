#!/usr/bin/env python3
"""
1d_PPO_Histogram_Zero_Cross_1wTrend_Filter
Hypothesis: The Percentage Price Oscillator (PPO) histogram crossing zero indicates momentum shifts.
Using 1d timeframe with 1w trend filter (EMA50) to capture major trend changes.
Volume confirmation ensures institutional participation. Designed for fewer trades (10-30/year)
to minimize fee drag in ranging 2025 market. Works in both bull (buy on bullish cross) and bear
(sell on bearish cross) markets.
"""

name = "1d_PPO_Histogram_Zero_Cross_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate PPO on daily data: (12 EMA - 26 EMA) / 26 EMA * 100
    ema12 = pd.Series(close).ewm(span=12, min_periods=12, adjust=False).mean().values
    ema26 = pd.Series(close).ewm(span=26, min_periods=26, adjust=False).mean().values
    ppo = (ema12 - ema26) / ema26 * 100
    
    # PPO histogram (same as PPO for simplicity, or could use signal line)
    ppo_hist = ppo  # Using PPO directly as histogram equivalent
    
    # Volume filter: 20-day EMA for spike detection
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # Fixed position size to minimize churn
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(ppo_hist[i]) or 
            np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # PPO histogram zero cross signals
        ppo_hist_prev = ppo_hist[i-1] if i > 0 else 0
        ppo_hist_cross_up = ppo_hist_prev <= 0 and ppo_hist[i] > 0
        ppo_hist_cross_down = ppo_hist_prev >= 0 and ppo_hist[i] < 0
        
        if position == 0:
            # Long: PPO crosses above zero + above weekly EMA50 + volume spike
            if ppo_hist_cross_up and close[i] > ema50_1w_aligned[i] and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: PPO crosses below zero + below weekly EMA50 + volume spike
            elif ppo_hist_cross_down and close[i] < ema50_1w_aligned[i] and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions - reverse signal or trend failure
            if position == 1:
                # Exit: PPO crosses below zero OR trend fails (close below weekly EMA)
                if ppo_hist_cross_down or close[i] < ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: PPO crosses above zero OR trend fails (close above weekly EMA)
                if ppo_hist_cross_up or close[i] > ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals