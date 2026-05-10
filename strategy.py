#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_WeeklyTrend_Volume
Hypothesis: Camarilla pivot levels on daily timeframe with weekly trend filter and volume confirmation.
Enters long when price breaks above R1 in uptrend (close > weekly EMA34), short when breaks below S1 in downtrend.
Uses volume spike (>1.5x weekly average volume) for confirmation. Exits when price returns to pivot point (PP).
Designed for low-frequency trading (10-25 trades/year) to minimize fee drag and work in both bull/bear markets.
"""

name = "1d_Camarilla_R1_S1_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
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
    
    # Calculate daily Camarilla pivot levels
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp = (high + low + close) / 3
    range_hl = high - low
    
    # Resistance and Support levels
    r1 = pp + (range_hl * 1.0 / 12)
    s1 = pp - (range_hl * 1.0 / 12)
    
    # Weekly EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        ema34_1w[33] = np.mean(close_1w[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1w)):
            ema34_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema34_1w[i-1]
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Weekly volume SMA10 for volume confirmation
    volume_1w = df_1w['volume'].values
    vol_sma10_1w = np.full(len(volume_1w), np.nan)
    if len(volume_1w) >= 10:
        vol_sma10_1w[9] = np.mean(volume_1w[:10])
        for i in range(10, len(volume_1w)):
            vol_sma10_1w[i] = (vol_sma10_1w[i-1] * 9 + volume_1w[i]) / 10
    vol_sma10_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_sma10_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Wait for weekly EMA34 to be ready
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_sma10_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current daily volume > 1.5x average weekly volume (scaled)
        # Approximate: weekly volume ~ 5x daily volume (5 trading days per week)
        vol_1w_scaled = vol_sma10_1w_aligned[i] / 5.0
        volume_confirm = volume[i] > 1.5 * vol_1w_scaled
        
        if position == 0:
            # Long: Price breaks above R1 in uptrend with volume confirmation
            if (close[i] > r1[i] and 
                close[i] > ema34_1w_aligned[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 in downtrend with volume confirmation
            elif (close[i] < s1[i] and 
                  close[i] < ema34_1w_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price returns to pivot point (PP) or trend reversal
            if (close[i] <= pp[i] or 
                close[i] < ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price returns to pivot point (PP) or trend reversal
            if (close[i] >= pp[i] or 
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals