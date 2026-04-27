#!/usr/bin/env python3
"""
6h_ElderRay_ZeroLine_Cross_1wTrend_VolumeSpike
Hypothesis: Uses Elder Ray Bull/Bear Power zero-line crosses on 6h timeframe for entries, filtered by 1-week EMA50 trend and volume spike (>1.8x average). Bull Power crossing above zero with 1w uptrend and volume confirmation triggers long entries; Bear Power crossing below zero with 1w downtrend and volume confirmation triggers short entries. Exits when the respective power crosses back through zero or trend breaks. Elder Ray measures buying/selling pressure relative to EMA13, providing early momentum signals. Weekly trend filter ensures alignment with higher timeframe momentum, reducing whipsaws. Volume confirmation ensures conviction. Designed for 6h timeframe with target 12-25 trades/year (~50-100 total over 4 years). Works in both bull and bear markets via 1-week trend filter and volume confirmation, capturing strong momentum moves while avoiding low-conviction noise.
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMA50 on 1w close for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Elder Ray on 6h timeframe: Bull Power = High - EMA13, Bear Power = Low - EMA13
    close_series = pd.Series(close)
    ema_13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Buying pressure
    bear_power = low - ema_13   # Selling pressure
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need EMA13 (13), 1w EMA50 (50), volume avg (20)
    start_idx = max(13, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        ema_1w_val = ema_50_1w_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Elder Ray zero-line cross with trend and volume confirmation
            # Long: Bull Power crosses above zero AND 1w uptrend AND volume confirmation
            # Use previous bar to detect cross
            if i > 0:
                bull_prev = bull_power[i-1]
                bear_prev = bear_power[i-1]
                bull_cross_up = (bull_prev <= 0) and (bull_val > 0)
                bear_cross_down = (bear_prev >= 0) and (bear_val < 0)
                
                if bull_cross_up and (close[i] > ema_1w_val) and vol_conf:
                    signals[i] = size
                    position = 1
                elif bear_cross_down and (close[i] < ema_1w_val) and vol_conf:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long when Bull Power crosses back below zero OR trend breaks
            if i > 0:
                bull_prev = bull_power[i-1]
                bull_cross_down = (bull_prev >= 0) and (bull_val < 0)
                trend_break = close[i] < ema_1w_val
                
                if bull_cross_down or trend_break:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = size
        elif position == -1:
            # Exit short when Bear Power crosses back above zero OR trend breaks
            if i > 0:
                bear_prev = bear_power[i-1]
                bear_cross_up = (bear_prev <= 0) and (bear_val > 0)
                trend_break = close[i] > ema_1w_val
                
                if bear_cross_up or trend_break:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -size
    
    return signals

name = "6h_ElderRay_ZeroLine_Cross_1wTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0