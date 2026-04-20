#!/usr/bin/env python3
# 6h_ElderRay_Trend_Momentum_Strategy
# Hypothesis: Elder Ray Index (Bull Power = High - EMA13, Bear Power = Low - EMA13) 
# captures institutional buying/selling pressure. Combined with 13-period EMA trend filter
# and volume confirmation, it identifies sustained momentum moves in both bull and bear markets.
# Uses 1-day EMA for multi-timeframe trend alignment to avoid false signals.
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.

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
    
    # Get 1-day data for multi-timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate EMA13 for Elder Ray (same timeframe)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # Buying pressure
    bear_power = low - ema13   # Selling pressure (negative values)
    
    # Calculate 1-day EMA34 for trend filter
    df_1d_close = pd.Series(df_1d['close'].values)
    ema34_1d = df_1d_close.ewm(span=34, adjust=False, min_periods=34).mean().values
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
            # Long: Bull Power > 0 (buying pressure) + price above 1D EMA34 + volume
            if bull_power[i] > 0 and close[i] > ema34_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (selling pressure) + price below 1D EMA34 + volume
            elif bear_power[i] < 0 and close[i] < ema34_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if bull power turns negative or price breaks below 1D EMA
            if bull_power[i] <= 0 or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if bear power turns positive or price breaks above 1D EMA
            if bear_power[i] >= 0 or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals