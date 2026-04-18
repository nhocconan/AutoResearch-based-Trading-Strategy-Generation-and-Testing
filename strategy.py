#!/usr/bin/env python3
"""
6h_ElderRay_RayEnergy_V1
Hypothesis: Elder Ray's Bull Power (high-EMA13) and Bear Power (low-EMA13) with 13-period EMA.
Go long when Bull Power > 0 AND Bear Power rising (less negative) AND 12h trend up (close > EMA34).
Go short when Bear Power < 0 AND Bull Power falling (less positive) AND 12h trend down (close < EMA34).
Requires volume > 1.3x 20-period average for confirmation.
Uses 6h timeframe targeting 15-25 trades/year (60-100 over 4 years).
Works in bull via trend-following longs and in bear via trend-following shorts.
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
    
    # EMA13 for Elder Ray (6h)
    ema_len = 13
    ema13 = np.full_like(close, np.nan)
    if len(close) >= ema_len:
        alpha = 2.0 / (ema_len + 1.0)
        ema13[0] = close[0]
        for i in range(1, len(close)):
            ema13[i] = alpha * close[i] + (1.0 - alpha) * ema13[i-1]
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema34_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 34:
        alpha_12h = 2.0 / (34 + 1.0)
        ema34_12h[0] = close_12h[0]
        for i in range(1, len(close_12h)):
            ema34_12h[i] = alpha_12h * close_12h[i] + (1.0 - alpha_12h) * ema34_12h[i-1]
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(ema_len, vol_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power rising (less negative) AND 12h uptrend
            if (bull_power[i] > 0 and bear_power[i] > bear_power[i-1] and 
                close[i] > ema34_12h_aligned[i] and vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power falling (less positive) AND 12h downtrend
            elif (bear_power[i] < 0 and bull_power[i] < bull_power[i-1] and 
                  close[i] < ema34_12h_aligned[i] and vol_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bear Power < 0 OR trend turns down
            if bear_power[i] < 0 or close[i] < ema34_12h_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bull Power > 0 OR trend turns up
            if bull_power[i] > 0 or close[i] > ema34_12h_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_RayEnergy_V1"
timeframe = "6h"
leverage = 1.0