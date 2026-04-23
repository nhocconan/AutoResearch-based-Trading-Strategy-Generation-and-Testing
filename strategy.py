#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 trend filter and volume confirmation
- Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 1d EMA13)
- Long when Bull Power > 0 AND close > 1d EMA50 AND volume > 1.5x 20-period average
- Short when Bear Power < 0 AND close < 1d EMA50 AND volume > 1.5x 20-period average
- Exit when power reverses (Bull Power <= 0 for long, Bear Power >= 0 for short)
- Uses 1d EMA50 for HTF trend alignment to avoid counter-trend entries
- Elder Ray measures buying/selling pressure relative to trend (EMA13)
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
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
    
    # Get 1d data for HTF indicators (EMA13 for power, EMA50 for trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray power calculation
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power (High - EMA13) and Bear Power (Low - EMA13)
    bull_power_1d = df_1d['high'].values - ema13_1d
    bear_power_1d = df_1d['low'].values - ema13_1d
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 1d indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for EMA50, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Elder Ray conditions
        bull_power_pos = bull_power_aligned[i] > 0  # Buying pressure
        bear_power_neg = bear_power_aligned[i] < 0  # Selling pressure
        
        # Trend filter (using 1d EMA50)
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0 + uptrend + volume confirmation
            if bull_power_pos and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 + downtrend + volume confirmation
            elif bear_power_neg and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Power reverses (loss of buying/selling pressure)
            exit_signal = False
            
            if position == 1:
                # Exit long: Bull Power <= 0 (buying pressure gone)
                if bull_power_aligned[i] <= 0:
                    exit_signal = True
            elif position == -1:
                # Exit short: Bear Power >= 0 (selling pressure gone)
                if bear_power_aligned[i] >= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_1dEMA50_Trend_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0