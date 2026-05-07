#!/usr/bin/env python3

name = "6h_ElderRay_ZoneRecovery_v2"
timeframe = "6h"
leverage = 1.0

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
    
    # Get daily data for Elder Ray and regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    # Calculate 13-day EMA for Elder Ray
    ema13 = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1d['high'].values - ema13
    bear_power = ema13 - df_1d['low'].values
    
    # Align Elder Ray to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 6-day EMA of bear power for signal line (Elder Ray signal)
    bear_power_ema6 = pd.Series(bear_power).ewm(span=6, adjust=False, min_periods=6).mean().values
    bear_power_ema6_aligned = align_htf_to_ltf(prices, df_1d, bear_power_ema6)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 20-week EMA for trend filter
    ema20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume filter: current volume > 1.5x 20-period average (6h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 48  # ~12 days for 6h to reduce trades
    
    start_idx = max(40, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or 
            np.isnan(bear_power_ema6_aligned[i]) or 
            np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine weekly trend direction
        weekly_up = ema20_1w_aligned[i] > ema20_1w_aligned[i-1] if i > 0 else False
        weekly_down = ema20_1w_aligned[i] < ema20_1w_aligned[i-1] if i > 0 else False
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Bull power positive AND rising, bear power below its EMA (recovery from selling)
            # In weekly uptrend with volume confirmation
            if (bull_power_aligned[i] > 0 and 
                bull_power_aligned[i] > bull_power_aligned[i-1] and  # Rising bull power
                bear_power_aligned[i] < bear_power_ema6_aligned[i] and  # Bear power below signal line
                weekly_up and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Bear power positive AND rising, bull power below zero (recovery from buying)
            # In weekly downtrend with volume confirmation
            elif (bear_power_aligned[i] > 0 and 
                  bear_power_aligned[i] > bear_power_aligned[i-1] and  # Rising bear power
                  bull_power_aligned[i] < 0 and  # Bull power negative
                  weekly_down and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Bull power turns negative OR bear power crosses above its EMA
            if (bull_power_aligned[i] <= 0) or (bear_power_aligned[i] > bear_power_ema6_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bear power turns negative OR bull power crosses above zero
            if (bear_power_aligned[i] <= 0) or (bull_power_aligned[i] >= 0):
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Elder Ray indicator with zone recovery signals on 6h timeframe.
# Long when bull power is positive AND rising (accumulation) while bear power is below its EMA
# (selling pressure weakening) in a weekly uptrend with volume confirmation.
# Short when bear power is positive AND rising (distribution) while bull power is negative
# (buying pressure weak) in a weekly downtrend with volume confirmation.
# Exits when the momentum shifts or the power dynamics reverse.
# Uses weekly EMA20 for trend filter to avoid counter-trend trades.
# Volume confirmation filters weak signals. Cooldown period reduces trade frequency.
# Works in both bull and bear markets by identifying shifts in bull/bear power dynamics.