#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme with 1d EMA50 trend filter and volume spike confirmation.
- Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100 over 14 periods
- Extreme readings: %R < -80 (oversold) or %R > -20 (overbought)
- Long: %R crosses above -80 from below (exit oversold) + volume > 2.0x 20-period avg + price > 1d EMA50 (uptrend)
- Short: %R crosses below -20 from above (exit overbought) + volume > 2.0x 20-period avg + price < 1d EMA50 (downtrend)
- Exit: Opposite extreme (%R crosses below -50 for long, above -50 for short) or trend reversal
- Uses 1d EMA50 for trend alignment to avoid counter-trend trades in strong trends
- Volume spike filter ensures participation, reducing false signals in low-volume environments
- Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag on 6h timeframe
- Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend)
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
    
    # Volume confirmation: > 2.0x 20-period average (strong spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    rr = highest_high - lowest_low
    williams_r = np.where(rr != 0, ((highest_high - close) / rr) * -100, -50)
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 50)  # Need 20 for volume MA, 14 for Williams %R, 50 for 1d EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 2.0x average)
        volume_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below (exit oversold) + volume spike + price > 1d EMA50 (uptrend)
            if i > start_idx:
                prev_williams_r = williams_r[i-1]
                curr_williams_r = williams_r[i]
                crossed_above_80 = prev_williams_r <= -80 and curr_williams_r > -80
            else:
                crossed_above_80 = False
            
            if volume_spike and close[i] > ema_50_aligned[i] and crossed_above_80:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above (exit overbought) + volume spike + price < 1d EMA50 (downtrend)
            elif i > start_idx:
                prev_williams_r = williams_r[i-1]
                curr_williams_r = williams_r[i]
                crossed_below_20 = prev_williams_r >= -20 and curr_williams_r < -20
            else:
                crossed_below_20 = False
                
            if volume_spike and close[i] < ema_50_aligned[i] and crossed_below_20:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 (loss of bullish momentum) or trend reversal
            if i > start_idx:
                prev_williams_r = williams_r[i-1]
                curr_williams_r = williams_r[i]
                crossed_below_50 = prev_williams_r >= -50 and curr_williams_r < -50
            else:
                crossed_below_50 = False
                
            if crossed_below_50 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 (loss of bearish momentum) or trend reversal
            if i > start_idx:
                prev_williams_r = williams_r[i-1]
                curr_williams_r = williams_r[i]
                crossed_above_50 = prev_williams_r <= -50 and curr_williams_r > -50
            else:
                crossed_above_50 = False
                
            if crossed_above_50 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0