#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrendFilter_VolumeConfirm_v1
Hypothesis: 12h Camarilla pivot breakout with daily trend filter and volume confirmation.
- Uses 12h timeframe for low trade frequency (target: 50-150 total trades over 4 years)
- Camarilla R1/S1 levels calculated from 1d OHLC (proven structure from DB)
- Long when price breaks above R1 with volume spike AND daily trend up (close > EMA34)
- Short when price breaks below S1 with volume spike AND daily trend down (close < EMA34)
- Volume confirmation: current volume > 1.5 * 20-period average
- Designed for 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
- Works in bull/bear markets by trading with the daily trend and using Camarilla for entry timing
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
    
    # Load daily data ONCE before loop for Camarilla levels, trend filter, and volume average
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from daily OHLC
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    camarilla_range = high_1d - low_1d
    r1 = close_1d + camarilla_range * 1.1 / 12
    s1 = close_1d - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (wait for completed daily bar)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily volume 20-period average for confirmation
    vol_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume average)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-day average
        volume_spike = volume[i] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Trend filter
        daily_uptrend = close[i] > ema34_1d_aligned[i]
        daily_downtrend = close[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 AND volume spike AND daily uptrend
            if close[i] > r1_aligned[i] and volume_spike and daily_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND volume spike AND daily downtrend
            elif close[i] < s1_aligned[i] and volume_spike and daily_downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below S1 (reversion to mean) OR daily trend changes
            if close[i] < s1_aligned[i] or not daily_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above R1 (reversion to mean) OR daily trend changes
            if close[i] > r1_aligned[i] or not daily_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrendFilter_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0