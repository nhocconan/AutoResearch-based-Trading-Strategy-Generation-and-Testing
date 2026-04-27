#!/usr/bin/env python3
"""
1d_Wide_Range_Breakout_WeeklyTrend_VolumeFilter
Hypothesis: Wide daily range breakouts (>1.5x ATR) aligned with weekly trend capture explosive moves in both bull and bear markets.
Weekly EMA50 filter avoids counter-trend trades. Volume >1.5x 20-day average confirms conviction.
Designed for low trade frequency (10-30/year) to minimize fee drag on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for daily ATR and weekly trend
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Daily ATR(14) for breakout threshold
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: >1.5x 20-day average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg_20)
    
    # Align indicators to daily timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need ATR (14), EMA50 (50), volume avg (20)
    start_idx = max(14, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_14_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        atr_val = atr_14_aligned[i]
        ema50 = ema50_1w_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        
        # Daily range
        daily_range = high[i] - low[i]
        
        if position == 0:
            # Determine weekly trend
            uptrend = close_val > ema50
            downtrend = close_val < ema50
            
            # Wide range breakout: daily range > 1.5x ATR
            wide_range = daily_range > (1.5 * atr_val)
            
            if uptrend and vol_conf and wide_range:
                # Long breakout: close above previous day's high
                if i > 0 and close_val > high[i-1]:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif downtrend and vol_conf and wide_range:
                # Short breakout: close below previous day's low
                if i > 0 and close_val < low[i-1]:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit: close below previous day's low or ATR-based stop
            if i > 0 and close_val < low[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: close above previous day's high or ATR-based stop
            if i > 0 and close_val > high[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Wide_Range_Breakout_WeeklyTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0