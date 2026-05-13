#!/usr/bin/env python3
"""
4h_Phase_Accumulation_Distribution_1dTrend_Volume
Hypothesis: Accumulation/distribution phases identified by price closing in upper/lower third of daily range combined with volume surge. In uptrend (price > 1d EMA50), go long on accumulation; in downtrend (price < 1d EMA50), go short on distribution. Volume confirmation filters low-quality signals. Designed for 4h timeframe to capture institutional accumulation/distribution phases in both bull and bear markets with low trade frequency.
"""

name = "4h_Phase_Accumulation_Distribution_1dTrend_Volume"
timeframe = "4h"
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
    
    # Get 1d data for accumulation/distribution phases
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily range and close position
    daily_range = df_1d['high'] - df_1d['low']
    close_position = (df_1d['close'] - df_1d['low']) / daily_range  # 0=low, 0.5=middle, 1=high
    
    # Accumulation: close in upper third (>0.67), Distribution: close in lower third (<0.33)
    accumulation = close_position > 0.67
    distribution = close_position < 0.33
    
    # Align accumulation/distribution signals to 4h timeframe
    accumulation_aligned = align_htf_to_ltf(prices, df_1d, accumulation.values.astype(float))
    distribution_aligned = align_htf_to_ltf(prices, df_1d, distribution.values.astype(float))
    
    # Get 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    cooldown = 0  # cooldown counter to prevent immediate re-entry
    
    for i in range(50, n):
        # Decrease cooldown if active
        if cooldown > 0:
            cooldown -= 1
        
        if position == 0 and cooldown == 0:
            # LONG: Accumulation phase with volume confirmation in uptrend
            if accumulation_aligned[i] > 0.5 and volume_confirmed[i] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Distribution phase with volume confirmation in downtrend
            elif distribution_aligned[i] > 0.5 and volume_confirmed[i] and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Distribution phase appears or trend weakens
            if distribution_aligned[i] > 0.5 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                cooldown = 2  # 2-bar cooldown after exit
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Accumulation phase appears or trend weakens
            if accumulation_aligned[i] > 0.5 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                cooldown = 2  # 2-bar cooldown after exit
            else:
                signals[i] = -0.25
    
    return signals