#!/usr/bin/env python3
"""
1d_Weekly_Momentum_With_Confirmation
Hypothesis: Combines weekly momentum (price vs 1-week ago) with volume confirmation and RSI filter on daily timeframe to capture sustained moves while avoiding whipsaws. Designed for low trade frequency (<25/year) and works in both bull and bear markets by following momentum with strict confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for momentum
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly momentum: current weekly close vs 1 week ago
    weekly_momentum = np.zeros_like(close_weekly)
    weekly_momentum[:] = np.nan
    for i in range(1, len(close_weekly)):
        weekly_momentum[i] = (close_weekly[i] - close_weekly[i-1]) / close_weekly[i-1]
    
    # Align weekly momentum to daily timeframe
    weekly_momentum_aligned = align_htf_to_ltf(prices, df_weekly, weekly_momentum)
    
    # Daily RSI for overbought/oversold filter (14-period)
    rsi = np.zeros_like(close)
    rsi[:] = np.nan
    if len(close) >= 14:
        # Calculate RSI using Wilder's smoothing
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        
        for i in range(14, len(close)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-20+1:i+1])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Warmup for RSI and volume
    
    for i in range(start_idx, n):
        if np.isnan(weekly_momentum_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: positive weekly momentum, RSI not overbought, volume spike
            if weekly_momentum_aligned[i] > 0.01 and rsi[i] < 70 and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: negative weekly momentum, RSI not oversold, volume spike
            elif weekly_momentum_aligned[i] < -0.01 and rsi[i] > 30 and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: weekly momentum turns negative OR RSI overbought
            if weekly_momentum_aligned[i] < 0 or rsi[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: weekly momentum turns positive OR RSI oversold
            if weekly_momentum_aligned[i] > 0 or rsi[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_Momentum_With_Confirmation"
timeframe = "1d"
leverage = 1.0