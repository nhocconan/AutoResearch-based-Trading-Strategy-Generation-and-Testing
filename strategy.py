#!/usr/bin/env python3
"""
4h_1d_DailyRangeBreakout_WithVolume_V1
Concept: Breakout of daily range with volume confirmation and EMA50 trend filter.
- Long: Price breaks above daily high + volume > 1.5x avg + price > EMA50
- Short: Price breaks below daily low + volume > 1.5x avg + price < EMA50
- Exit: Price crosses EMA50 in opposite direction (trend reversal)
- Uses daily levels for structure, EMA50 for trend filter, volume for confirmation
- Designed to work in both trending and ranging markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_DailyRangeBreakout_WithVolume_V1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Daily range: previous day's high and low ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use previous day's values (shift by 1) to avoid look-ahead
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    # Align daily levels to 4h timeframe
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    daily_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close_1d)
    
    # === 4h indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA50 trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA50
    
    for i in range(start_idx, n):
        # Get values
        ema50_val = ema50[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        daily_high = daily_high_aligned[i]
        daily_low = daily_low_aligned[i]
        daily_close = daily_close_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema50_val) or np.isnan(daily_high) or np.isnan(daily_low) or 
            np.isnan(daily_close) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above daily high with volume confirmation and above EMA50
            breakout_long = high_val > daily_high
            vol_confirm = vol_ratio_val > 1.5
            
            if breakout_long and vol_confirm and close_val > ema50_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below daily low with volume confirmation and below EMA50
            elif low_val < daily_low and vol_confirm and close_val < ema50_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below EMA50 (trend reversal)
            if close_val < ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above EMA50 (trend reversal)
            if close_val > ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals