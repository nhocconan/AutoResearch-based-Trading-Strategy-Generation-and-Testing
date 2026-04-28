#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_Volume
Hypothesis: Elder Ray's Bull/Bear Power combined with 1-day trend and volume filter.
Works in both bull and bear by trading with the higher timeframe trend.
Targets 20-50 trades/year to minimize fee drag.
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
    
    # Get 1-day data for trend filter and EMA13 for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate EMA13 for Elder Ray (on 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume filter: >1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_13[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction from 1-day EMA34
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = vol_filter[i]
        
        # Elder Ray signals
        bull_signal = bull_power[i] > 0 and bull_power[i] > bull_power[i-1]
        bear_signal = bear_power[i] < 0 and bear_power[i] < bear_power[i-1]
        
        # Entry logic:
        # Long: Bull power rising + uptrend + volume
        long_entry = vol_confirm and trend_up and bull_signal
        # Short: Bear power falling + downtrend + volume
        short_entry = vol_confirm and trend_down and bear_signal
        
        # Exit logic: Opposite Elder Ray signal or trend reversal
        long_exit = bear_signal or not trend_up
        short_exit = bull_signal or not trend_down
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0