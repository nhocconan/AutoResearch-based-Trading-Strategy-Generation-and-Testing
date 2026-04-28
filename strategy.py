#!/usr/bin/env python3
"""
1d_4hTrend_WeeklyVolumeBreakout
Hypothesis: Uses 4h EMA20 trend and 1w volume spike to filter breakouts on daily timeframe.
Enters long when price breaks above 4h EMA20 with weekly volume spike, short when breaks below.
Designed to capture trend momentum while filtering false signals in both bull and bear markets.
Targets 15-25 trades per year to minimize fee decay.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA20 trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA20
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Get 1w data for volume spike
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w volume 20-period MA
    vol_1w = df_1w['volume'].values
    vol_ma_20_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    
    # Volume spike: current 1d volume > 2.0x 1w volume MA
    vol_spike = volume > (2.0 * vol_ma_20_1w_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(vol_ma_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction from 4h EMA20
        price_above_ema = close[i] > ema_20_4h_aligned[i]
        price_below_ema = close[i] < ema_20_4h_aligned[i]
        
        # Volume confirmation
        vol_confirm = vol_spike[i]
        
        # Entry logic:
        # Long: price above EMA20 with volume spike
        long_entry = price_above_ema and vol_confirm
        # Short: price below EMA20 with volume spike
        short_entry = price_below_ema and vol_confirm
        
        # Exit logic: price crosses back through EMA20
        long_exit = price_below_ema
        short_exit = price_above_ema
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_4hTrend_WeeklyVolumeBreakout"
timeframe = "1d"
leverage = 1.0