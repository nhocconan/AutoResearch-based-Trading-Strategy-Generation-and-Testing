#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ElderRay_ForceIndex_Breakout"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # === Daily Elder Ray Index ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # EMA13 of daily high/low for Bull/Bear Power
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    ema13_high = high_series.ewm(span=13, min_periods=13, adjust=False).mean().values
    ema13_low = low_series.ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Bull Power = High - EMA13(High), Bear Power = EMA13(Low) - Low
    bull_power = high_1d - ema13_high
    bear_power = ema13_low - low_1d
    
    # Align to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # === 6h Force Index (EMA13 of price*volume change) ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Price change * volume
    price_change = np.diff(close, prepend=close[0])
    raw_force = price_change * volume
    
    # EMA13 of Force Index
    force_series = pd.Series(raw_force)
    force_index = force_series.ewm(span=13, min_periods=13, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):
        # Get values
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        force_val = force_index[i]
        
        # Skip if any value is NaN
        if (np.isnan(bull_val) or np.isnan(bear_val) or np.isnan(force_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (strong) AND Force Index turning up from negative
            if bull_val > 0 and force_val > 0 and (i == 13 or force_index[i-1] <= 0):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 (strong) AND Force Index turning down from positive
            elif bear_val > 0 and force_val < 0 and (i == 13 or force_index[i-1] >= 0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bull Power turns negative OR Force Index turns down
            if bull_val <= 0 or (i > 13 and force_val < 0 and force_index[i-1] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bear Power turns negative OR Force Index turns up
            if bear_val <= 0 or (i > 13 and force_val > 0 and force_index[i-1] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals