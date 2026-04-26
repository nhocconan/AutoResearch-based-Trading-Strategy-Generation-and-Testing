#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrendFilter_VolumeConfirm_v1
Hypothesis: Daily Camarilla pivot breakout with weekly trend filter and volume confirmation.
- Uses 1d timeframe for low trade frequency (target: 30-100 total trades over 4 years)
- Camarilla R1/S1 levels calculated from previous 1d bar
- Long when price breaks above R1 with volume spike and weekly uptrend
- Short when price breaks below S1 with volume spike and weekly downtrend
- Designed for 7-25 trades/year (30-100 total over 4 years) to minimize fee drag
- Weekly trend filter prevents counter-trend trades in bear markets (2022 crash, 2025-2026 range)
- Volume confirmation ensures breakout validity
- Works in bull/bear markets by trading with weekly trend and using Camarilla for precise entries
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
    
    # Load weekly data ONCE before loop for trend filter and Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Camarilla levels from previous weekly bar
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    
    camarilla_range = (high_1w - low_1w) * 1.1 / 12
    r1_1w = close_1w_arr + camarilla_range
    s1_1w = close_1w_arr - camarilla_range
    
    # Align Camarilla levels to daily timeframe (wait for completed weekly bar)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Calculate volume spike (20-period volume average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA, 50 for EMA)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla breakout conditions with volume confirmation
        price_above_r1 = close[i] > r1_aligned[i]
        price_below_s1 = close[i] < s1_aligned[i]
        
        # Weekly trend filter
        trend_up = close[i] > ema50_1w_aligned[i]
        trend_down = close[i] < ema50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 AND volume spike AND weekly uptrend
            if price_above_r1 and volume_spike[i] and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND volume spike AND weekly downtrend
            elif price_below_s1 and volume_spike[i] and trend_down:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below S1 OR weekly trend turns down
            if price_below_s1 or not trend_up:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above R1 OR weekly trend turns up
            if price_above_r1 or not trend_down:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrendFilter_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0