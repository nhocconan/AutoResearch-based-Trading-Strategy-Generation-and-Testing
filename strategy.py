#!/usr/bin/env python3
"""
1d_CAMARILLA_R1S1_Volume_Confirm
1d strategy trading CAMARILLA pivot R1/S1 breakouts with volume confirmation.
- Long: Close breaks above R1 + volume > 1.5x daily average volume
- Short: Close breaks below S1 + volume > 1.5x daily average volume
- Exit: Close reverts back inside the R1-S1 range
Designed for 10-25 trades/year per symbol (40-100 total over 4 years)
Uses weekly trend filter to avoid counter-trend trades in strong trends
"""

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get daily data for CAMARILLA pivots and volume average
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate CAMARILLA pivot levels for each day
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    pivot_range = (high_1d - low_1d) * 1.1 / 12
    r1_level = close_1d + pivot_range
    s1_level = close_1d - pivot_range
    
    # Align R1/S1 levels to 1d timeframe (they're already daily)
    r1_aligned = r1_level  # No alignment needed for same timeframe
    s1_aligned = s1_level
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = vol_ma_20  # Already aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # need enough for weekly EMA34
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter
        weekly_uptrend = ema_34_1w_aligned[i] > close_1d[i]  # price above weekly EMA34 = uptrend
        weekly_downtrend = ema_34_1w_aligned[i] < close_1d[i]  # price below weekly EMA34 = downtrend
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Breakout conditions
        breakout_r1 = close[i] > r1_aligned[i]
        breakdown_s1 = close[i] < s1_aligned[i]
        
        # Mean reversion exit: price returns inside R1-S1 range
        inside_range = (close[i] >= s1_aligned[i]) and (close[i] <= r1_aligned[i])
        
        if position == 0:
            # Long: weekly uptrend + volume + breakout above R1
            if weekly_uptrend and vol_confirm and breakout_r1:
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + volume + breakdown below S1
            elif weekly_downtrend and vol_confirm and breakdown_s1:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: weekly trend turns down OR price returns inside range
            if not weekly_uptrend or inside_range:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: weekly trend turns up OR price returns inside range
            if not weekly_downtrend or inside_range:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_CAMARILLA_R1S1_Volume_Confirm"
timeframe = "1d"
leverage = 1.0