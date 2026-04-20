#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_RangeBreakout_Volume_V1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily range (high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Previous day's range for breakout level
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_volume = np.roll(volume_1d, 1)
    
    # Range breakout levels (using previous day's high/low)
    range_high = prev_high
    range_low = prev_low
    
    # Align daily levels to 12h timeframe
    range_high_aligned = align_htf_to_ltf(prices, df_1d, range_high)
    range_low_aligned = align_htf_to_ltf(prices, df_1d, range_low)
    prev_volume_aligned = align_htf_to_ltf(prices, df_1d, prev_volume)
    
    # 12h indicators
    close = prices['close'].values
    volume = prices['volume'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Volume spike detection (current volume vs previous day's volume)
    volume_spike = volume / np.where(prev_volume_aligned > 0, prev_volume_aligned, np.nan)
    
    # 12h ATR for volatility filter (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 12h EMA for trend filter (20-period)
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_spike_val = volume_spike[i]
        atr_val = atr[i]
        ema20_val = ema20[i]
        range_high_val = range_high_aligned[i]
        range_low_val = range_low_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_spike_val) or np.isnan(atr_val) or np.isnan(ema20_val) or 
            np.isnan(range_high_val) or np.isnan(range_low_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above previous day's high with volume spike and above EMA
            if (high_val > range_high_val and 
                vol_spike_val > 2.0 and
                close_val > ema20_val):
                signals[i] = 0.25
                position = 1
            # Short: Break below previous day's low with volume spike and below EMA
            elif (low_val < range_low_val and 
                  vol_spike_val > 2.0 and
                  close_val < ema20_val):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below previous day's low or volatility drops
            if low_val < range_low_val or vol_spike_val < 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above previous day's high or volatility drops
            if high_val > range_high_val or vol_spike_val < 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals