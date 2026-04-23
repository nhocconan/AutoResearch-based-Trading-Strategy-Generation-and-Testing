#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA50 trend filter and volume confirmation
- Long when Bull Power > 0 (close > EMA13) AND Bear Power < 0 (close < EMA13) AND price > 1d EMA50 AND volume > 1.5x 20-period average
- Short when Bear Power < 0 (close < EMA13) AND Bull Power > 0 (close > EMA13) AND price < 1d EMA50 AND volume > 1.5x 20-period average
- Exit when Elder Ray signals reverse (Bull Power <= 0 for longs, Bear Power >= 0 for shorts)
- Uses 1d EMA50 for HTF trend alignment to avoid counter-trend trades
- Volume confirmation ensures institutional participation
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
- Works in both bull and bear markets by following the 1d trend while using Elder Ray for precise entries
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
    
    # Get 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Elder Ray Index (Bull Power and Bear Power) on 6h data
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power: high minus EMA13
    bear_power = low - ema13   # Bear Power: low minus EMA13
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 51)  # Need 20 for volume MA, 14 for EMA13, 51 for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Elder Ray conditions
        bull_power_positive = bull_power[i] > 0  # Bullish momentum
        bear_power_negative = bear_power[i] < 0  # Bearish momentum
        
        # Trend filter (using 1d EMA50)
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: bullish momentum + uptrend + volume confirmation
            if bull_power_positive and bear_power_negative and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: bearish momentum + downtrend + volume confirmation
            elif bear_power_negative and bull_power_positive and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Elder Ray signals reverse
            exit_signal = False
            
            if position == 1:
                # Exit long: bullish momentum disappears (Bull Power <= 0)
                if bull_power[i] <= 0:
                    exit_signal = True
            elif position == -1:
                # Exit short: bearish momentum disappears (Bear Power >= 0)
                if bear_power[i] >= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_1dEMA50_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0