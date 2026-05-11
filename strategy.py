#!/usr/bin/env python3
"""
1d_1w_WeeklyHighLow_Breakout_Trend_Confirmation
Hypothesis: Breakout of weekly high/low levels on daily timeframe with weekly trend filter.
Weekly trend provides directional bias, daily breakout provides entry timing. Works in both bull and bear markets by aligning with higher timeframe trend.
Target: 20-40 trades per year to minimize fee drag while capturing significant moves.
"""

name = "1d_1w_WeeklyHighLow_Breakout_Trend_Confirmation"
timeframe = "1d"
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
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # === 1w Data (loaded ONCE) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === Weekly High/Low Levels ===
    # Use previous week's high/low for breakout (already completed)
    weekly_high = high_1w
    weekly_low = low_1w
    
    # Align weekly levels to daily
    weekly_high_daily = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_daily = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # === Weekly Trend Filter (EMA 21) ===
    ema21_1w = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema21_1w_daily = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # === Volume Filter (20-day EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5  # Require 1.5x average volume
    
    # === Signal Parameters ===
    position_size = 0.25  # 25% of capital per trade
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    holding_bars = 0
    
    # Start after warmup (covers weekly EMA21)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(weekly_high_daily[i]) or np.isnan(weekly_low_daily[i]) or 
            np.isnan(ema21_1w_daily[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                holding_bars = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Break above weekly high + above weekly EMA21 + volume spike
            if (open_price[i] > weekly_high_daily[i] and close[i] > weekly_high_daily[i] and 
                close[i] > ema21_1w_daily[i] and volume_ok[i]):
                signals[i] = position_size
                position = 1
                holding_bars = 0
            # Short: Break below weekly low + below weekly EMA21 + volume spike
            elif (open_price[i] < weekly_low_daily[i] and close[i] < weekly_low_daily[i] and 
                  close[i] < ema21_1w_daily[i] and volume_ok[i]):
                signals[i] = -position_size
                position = -1
                holding_bars = 0
        else:
            # Enforce minimum holding period (5 days)
            holding_bars += 1
            if holding_bars < 5:
                signals[i] = position_size if position == 1 else -position_size
                continue
            
            # Exit: Price closes below/above opposite level
            if position == 1:
                if close[i] < weekly_low_daily[i]:
                    signals[i] = 0.0
                    position = 0
                    holding_bars = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if close[i] > weekly_high_daily[i]:
                    signals[i] = 0.0
                    position = 0
                    holding_bars = 0
                else:
                    signals[i] = -position_size
    
    return signals