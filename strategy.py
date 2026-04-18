#!/usr/bin/env python3
"""
6h Elder Ray (Bull/Bear Power) + Volume Spike + Regime Filter
Strategy: Elder Ray measures bull/bear power via EMA13. 
Long when bull power > 0, volume spike, and price above weekly EMA50.
Short when bear power < 0, volume spike, and price below weekly EMA50.
Uses daily EMA13 for Elder Ray and weekly EMA50 for trend filter.
Designed to work in both bull and bear markets by following institutional flow.
"""

import numpy as np
import pandas as pd
from mtd_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Elder Ray (EMA13)
    df_1d = get_htf_data(prices, '1d')
    # Get weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate daily EMA13 for Elder Ray
    ema_13_1d = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    # Bull Power = High - EMA13
    bull_power = high - ema_13_1d
    # Bear Power = Low - EMA13
    bear_power = low - ema_13_1d
    
    # Calculate weekly EMA50 for trend filter
    weekly_close = df_1w['close'].values
    ema_50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Align weekly EMA50 to 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        ema_50_weekly = ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: bull power positive, volume spike, price above weekly EMA50
            if (bull_power_val > 0 and volume_spike[i] and price > ema_50_weekly):
                signals[i] = 0.25
                position = 1
            # Short: bear power negative, volume spike, price below weekly EMA50
            elif (bear_power_val < 0 and volume_spike[i] and price < ema_50_weekly):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: bull power turns negative or price below weekly EMA50
            if bull_power_val <= 0 or price < ema_50_weekly:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: bear power turns positive or price above weekly EMA50
            if bear_power_val >= 0 or price > ema_50_weekly:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_VolumeSpike_WeeklyEMA50"
timeframe = "6h"
leverage = 1.0