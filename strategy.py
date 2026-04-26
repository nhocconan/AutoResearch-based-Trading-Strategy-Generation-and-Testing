#!/usr/bin/env python3
"""
1d_WeeklyDonchianBreakout_WeeklyTrend_v1
Hypothesis: Daily Donchian(20) breakout with weekly trend filter and volume confirmation.
- Uses 1d timeframe for low trade frequency (target: 30-100 total trades over 4 years)
- Donchian channels calculated from previous day's high/low
- Long when price breaks above upper channel with volume spike and weekly uptrend
- Short when price breaks below lower channel with volume spike and weekly downtrend
- Weekly trend filter uses EMA34 on weekly closes for stability
- Designed for 7-25 trades/year (30-100 total over 4 years) to minimize fee drag
- Works in bull/bear markets by trading with weekly trend and using Donchian for structure
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Donchian channels from previous day
    # Upper channel = 20-period high, Lower channel = 20-period low
    high_ma20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian, 34 for weekly EMA)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if weekly trend not ready
        if np.isnan(ema34_1w_aligned[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Donchian breakout conditions
        price_above_upper = close[i] > high_ma20[i-1]  # Previous day's upper channel
        price_below_lower = close[i] < low_ma20[i-1]   # Previous day's lower channel
        
        # Volume confirmation (20-period volume average)
        if i >= 20:
            vol_ma20 = np.mean(volume[i-20:i])
            volume_spike = volume[i] > (vol_ma20 * 2.0)
        else:
            volume_spike = False
        
        # Weekly trend filter
        trend_up = close[i] > ema34_1w_aligned[i]
        trend_down = close[i] < ema34_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper channel AND volume spike AND weekly uptrend
            if price_above_upper and volume_spike and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel AND volume spike AND weekly downtrend
            elif price_below_lower and volume_spike and trend_down:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below lower channel OR weekly trend turns down
            if price_below_lower or not trend_up:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above upper channel OR weekly trend turns up
            if price_above_upper or not trend_down:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WeeklyDonchianBreakout_WeeklyTrend_v1"
timeframe = "1d"
leverage = 1.0