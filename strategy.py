#!/usr/bin/env python3
"""
1h_4h_1d_SupportResistance_Breakout_VolumeTrendFilter_v1
Concept: Use daily support/resistance levels with 4h trend filter and volume confirmation.
- Long when price breaks above prior day's high with volume > 2x average and above 4h EMA34
- Short when price breaks below prior day's low with volume > 2x average and below 4h EMA34
- Exit when price returns to prior day's close (mean reversion to daily value)
- Session filter: Only trade 08-20 UTC to avoid low-volume Asian session
- Conservative sizing: 0.20 to manage drawdown
- Designed for breakouts in bull markets and mean reversion in bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_SupportResistance_Breakout_VolumeTrendFilter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for barriers
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Calculate daily barriers: previous day's high, low, close ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use previous day's values (shift by 1) to avoid look-ahead
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    # Set first day's values to NaN (no previous day)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    # Align barriers to 1h timeframe
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close_1d)
    
    # === 4h: EMA34 trend filter ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # === 1h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Precompute session hours (08-20 UTC)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA and volume MA
    
    for i in range(start_idx, n):
        # Session filter: Only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        ema34_val = ema34_4h_aligned[i]
        close_val = prices['close'].iloc[i]
        high_val = prev_high_aligned[i]
        low_val = prev_low_aligned[i]
        close_barrier_val = prev_close_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema34_val) or np.isnan(high_val) or np.isnan(low_val) or 
            np.isnan(close_barrier_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above prior day's high with volume confirmation and above 4h EMA34
            breakout_long = close_val > high_val
            vol_confirm = vol_ratio_val > 2.0
            
            if breakout_long and vol_confirm and close_val > ema34_val:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below prior day's low with volume confirmation and below 4h EMA34
            elif close_val < low_val and vol_confirm and close_val < ema34_val:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to or below prior day's close
            if close_val <= close_barrier_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: Price returns to or above prior day's close
            if close_val >= close_barrier_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals