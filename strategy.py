#!/usr/bin/env python3
# 6h_ElderRay_Trend_Momentum_Strategy
# Hypothesis: Elder Ray Index (Bull/Bear Power) combined with EMA13 trend filter and volume confirmation.
# Bull Power = High - EMA13, Bear Power = Low - EMA13. Long when Bull Power > 0 and rising, Bear Power < 0.
# Short when Bear Power < 0 and falling, Bull Power > 0. Uses 1d trend filter for multi-timeframe alignment.
# Designed to capture momentum in both bull and bear markets by measuring bull/bear strength relative to trend.
# Target: 20-40 trades/year.

name = "6h_ElderRay_Trend_Momentum_Strategy"
timeframe = "6h"
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
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = low - ema13   # Low - EMA13
    
    # 1d EMA34 for trend filter
    df_1d_close = df_1d['close'].values
    ema34_1d = pd.Series(df_1d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure EMA34 and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 and rising, Bear Power < 0, above 1d EMA34, volume confirmation
            if (bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and 
                bear_power[i] < 0 and close[i] > ema34_1d_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 and falling, Bull Power > 0, below 1d EMA34, volume confirmation
            elif (bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and 
                  bull_power[i] > 0 and close[i] < ema34_1d_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if Bull Power <= 0 or breaks below 1d EMA34
            if bull_power[i] <= 0 or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if Bear Power >= 0 or breaks above 1d EMA34
            if bear_power[i] >= 0 or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals