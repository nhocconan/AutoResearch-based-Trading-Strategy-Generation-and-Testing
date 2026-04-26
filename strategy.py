#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_WeeklyTrend_VolumeConfirm_v1
Hypothesis: Elder Ray Index (Bull/Bear Power) with 13-period EMA on 6h, filtered by weekly trend (price vs weekly EMA50) and volume confirmation (>1.5x average volume). Bull Power > 0 indicates buying pressure, Bear Power < 0 indicates selling pressure. Weekly trend filter ensures we trade with the higher timeframe momentum. Designed for 6h to target 12-37 trades/year with discrete sizing (0.25).
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
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 13-period EMA for Elder Ray (on 6h close)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13
    
    # Average volume for confirmation (24-period SMA = 4h * 2 = 8h, approx 1/3 day)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of weekly EMA(50), EMA(13), volume(24)
    start_idx = max(50, 13, 24)
    
    for i in range(start_idx, n):
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_13_val = ema_13[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        ema_weekly_val = ema_50_1w_aligned[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_13_val) or np.isnan(avg_vol) or np.isnan(bull_val) or 
            np.isnan(bear_val) or np.isnan(ema_weekly_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Weekly trend filter
        uptrend = close_val > ema_weekly_val
        downtrend = close_val < ema_weekly_val
        
        # Long: Bull Power > 0 (buying pressure) + weekly uptrend + volume
        long_condition = (bull_val > 0) and uptrend and volume_confirmed
        # Short: Bear Power < 0 (selling pressure) + weekly downtrend + volume
        short_condition = (bear_val < 0) and downtrend and volume_confirmed
        
        # Exit: opposite Elder Ray signal or loss of weekly trend alignment
        long_exit = (position == 1 and (bear_val >= 0 or not uptrend))
        short_exit = (position == -1 and (bull_val <= 0 or not downtrend))
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_ElderRay_BullBearPower_WeeklyTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0