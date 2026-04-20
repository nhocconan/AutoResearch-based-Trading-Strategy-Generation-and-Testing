#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_EMA20_Trend_V2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d: Elder Ray components (EMA20) ===
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA20 for Elder Ray
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Bull Power and Bear Power
    bull_power = high_1d - ema20_1d
    bear_power = low_1d - ema20_1d
    
    # 13-period EMA of Bull/Bear Power for smoothing
    ema13_bull = pd.Series(bull_power).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_bear = pd.Series(bear_power).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Align to 6h timeframe
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    ema13_bull_aligned = align_htf_to_ltf(prices, df_1d, ema13_bull)
    ema13_bear_aligned = align_htf_to_ltf(prices, df_1d, ema13_bear)
    
    # === 6h: Price and volume ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip outside session
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_val = ema20_1d_aligned[i]
        bull_val = ema13_bull_aligned[i]
        bear_val = ema13_bear_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(ema_val) or np.isnan(bull_val) or np.isnan(bear_val) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull power positive and rising + price above EMA20 + volume
            if (bull_val > 0 and                  # Bull power positive
                bull_val > ema13_bull_aligned[i-1] and  # Bull power rising
                close_val > ema_val and             # Price above EMA20
                vol_ratio_val > 1.3):               # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Bear power negative and falling + price below EMA20 + volume
            elif (bear_val < 0 and                # Bear power negative
                  bear_val < ema13_bear_aligned[i-1] and  # Bear power falling (more negative)
                  close_val < ema_val and             # Price below EMA20
                  vol_ratio_val > 1.3):               # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bear power turns positive or volume dries up
            if (bear_val > 0 or                   # Bear power turns positive (bulls losing)
                vol_ratio_val < 0.7):             # Low volume (losing momentum)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bull power turns negative or volume dries up
            if (bull_val < 0 or                   # Bull power turns negative (bears losing)
                vol_ratio_val < 0.7):             # Low volume (losing momentum)
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals