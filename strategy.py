#!/usr/bin/env python3
"""
6h_ElderRay_Breakout_12hTrend_VolumeConfirm
Hypothesis: 6h Elder Ray (Bull/Bear Power) breakout with 12h EMA50 trend filter and volume confirmation (>1.8x median). Enters long when Bull Power turns positive with price above 12h EMA50 and volume spike. Enters short when Bear Power turns negative with price below 12h EMA50 and volume spike. Uses discrete sizing (0.25) to minimize churn. Target: 50-150 trades over 4 years. Works in bull/bear via 12h trend filter and volume confirmation to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Elder Ray on 6h: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Load 12h data for HTF trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 1.8x 50-period median
    volume_series = pd.Series(volume)
    vol_median = volume_series.rolling(window=50, min_periods=50).median().values
    volume_confirm = volume > (1.8 * vol_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 13-period EMA, 50-period EMA, 50-period volume median)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_median[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: Bull Power turns positive + price above 12h EMA50 + volume confirmation
        long_signal = (bull_power[i] > 0 and bull_power[i-1] <= 0) and \
                      (close[i] > ema_50_12h_aligned[i]) and \
                      volume_confirm[i]
        
        # Short logic: Bear Power turns negative + price below 12h EMA50 + volume confirmation
        short_signal = (bear_power[i] < 0 and bear_power[i-1] >= 0) and \
                       (close[i] < ema_50_12h_aligned[i]) and \
                       volume_confirm[i]
        
        # Exit logic: opposite Elder Ray signal
        exit_long = position == 1 and (bear_power[i] < 0 and bear_power[i-1] >= 0)
        exit_short = position == -1 and (bull_power[i] > 0 and bull_power[i-1] <= 0)
        
        if long_signal:
            if position != 1:
                signals[i] = base_size
                position = 1
            else:
                signals[i] = base_size
        elif short_signal:
            if position != -1:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = -base_size
        elif exit_long or exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_ElderRay_Breakout_12hTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0