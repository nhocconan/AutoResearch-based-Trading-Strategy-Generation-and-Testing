#!/usr/bin/env python3
"""
1D_Weekly_Keltner_Breakout_Trend_Filter_v1
Hypothesis: Use daily price breaks above/below Keltner Channel (20,1.5) with weekly trend filter.
Long when daily close crosses above upper Keltner and weekly close > weekly EMA20.
Short when daily close crosses below lower Keltner and weekly close < weekly EMA20.
Volume confirmation: current volume > 1.3x 20-day average volume.
This strategy targets breakouts with trend alignment and volume confirmation to work in both bull and bear markets.
"""
name = "1D_Weekly_Keltner_Breakout_Trend_Filter_v1"
timeframe = "1d"
leverage = 1.0

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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend
    close_weekly = pd.Series(df_weekly['close'])
    ema_weekly = close_weekly.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Calculate daily Keltner Channel (20, 1.5)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Keltner Channel
    ema_close = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    keltner_upper = ema_close + (1.5 * atr)
    keltner_lower = ema_close - (1.5 * atr)
    
    # Volume filter: current volume > 1.3 * 20-day average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = 20  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(ema_weekly_aligned[i]) or np.isnan(keltner_upper[i]) or 
            np.isnan(keltner_lower[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 5 days between trades to reduce frequency
            if bars_since_exit < 5:
                continue
                
            # Long: daily close crosses above upper Keltner and weekly trend up
            if (close[i] > keltner_upper[i] and close[i-1] <= keltner_upper[i-1] and 
                close[i] > ema_weekly_aligned[i] and volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: daily close crosses below lower Keltner and weekly trend down
            elif (close[i] < keltner_lower[i] and close[i-1] >= keltner_lower[i-1] and 
                  close[i] < ema_weekly_aligned[i] and volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: daily close returns to opposite EMA
            if position == 1 and close[i] < ema_close[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and close[i] > ema_close[i]:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals