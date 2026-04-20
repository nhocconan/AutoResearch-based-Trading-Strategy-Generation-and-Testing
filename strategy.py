#!/usr/bin/env python3
# 6h_ElderRay_Trend_Momentum_Strategy
# Hypothesis: Elder Ray (Bull Power/Bear Power) combined with EMA trend filter and volume confirmation
# captures trend momentum in both bull and bear markets. Bull Power > 0 and Bear Power < 0
# indicate bullish/bearish strength relative to EMA. Works across regimes by requiring
# alignment of power, trend, and volume. Target: 20-40 trades/year (80-160 total over 4 years).

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
    
    # Get daily data for Elder Ray and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate EMA(13) for Elder Ray (standard setting)
    close_1d = df_1d['close'].values
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema_13  # Bull Power = High - EMA
    bear_power = low_1d - ema_13   # Bear Power = Low - EMA
    
    # Align to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13)
    
    # Additional trend filter: EMA(34) for direction
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure EMA(34) is calculated
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema_13_aligned[i]) or np.isnan(ema_34_aligned[i]) or
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (strength) + price > EMA(34) (uptrend) + volume
            if bull_power_aligned[i] > 0 and close[i] > ema_34_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (weakness) + price < EMA(34) (downtrend) + volume
            elif bear_power_aligned[i] < 0 and close[i] < ema_34_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if Bear Power >= 0 (loss of strength) or price < EMA(13)
            if bear_power_aligned[i] >= 0 or close[i] < ema_13_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if Bull Power <= 0 (loss of weakness) or price > EMA(13)
            if bull_power_aligned[i] <= 0 or close[i] > ema_13_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals