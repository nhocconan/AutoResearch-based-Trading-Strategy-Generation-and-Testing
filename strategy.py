#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Elder Ray Index with 1d trend filter and volume spike.
# Long when Elder Ray Bull Power > 0 AND 1d EMA50 trending up AND 1d volume spike.
# Short when Elder Ray Bear Power < 0 AND 1d EMA50 trending down AND 1d volume spike.
# Uses Elder Ray to measure bull/bear power relative to EMA, with 1d trend and volume for confirmation.
# Designed for low trade frequency (target: 15-25/year) to minimize fee drag and improve generalization.
# Works in bull markets via Bull Power and in bear markets via Bear Power.
name = "12h_ElderRay_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA50 trend and volume spike
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # 1d EMA50 for trend
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_prev = np.roll(ema50_1d, 1)
    ema50_1d_prev[0] = ema50_1d[0]
    ema50_up = ema50_1d > ema50_1d_prev
    ema50_down = ema50_1d < ema50_1d_prev
    ema50_up_aligned = align_htf_to_ltf(prices, df_1d, ema50_up)
    ema50_down_aligned = align_htf_to_ltf(prices, df_1d, ema50_down)
    
    # 1d volume spike: current volume > 2.0 * 20-period EMA
    vol_ema_20 = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike_1d = np.where(vol_ema_20 > 0, df_1d['volume'].values / vol_ema_20, 1.0) > 2.0
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Load 12h data for Elder Ray Index (13-period EMA)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA13 for Elder Ray
    close_12h = df_12h['close'].values
    ema13_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13_12h
    bear_power = low - ema13_12h
    
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Sufficient warmup for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema50_up_aligned[i]) or np.isnan(ema50_down_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long condition: Bull Power > 0, 1d EMA50 trending up, volume spike
            long_condition = (bull_power_aligned[i] > 0) and ema50_up_aligned[i] and vol_spike_1d_aligned[i]
            # Short condition: Bear Power < 0, 1d EMA50 trending down, volume spike
            short_condition = (bear_power_aligned[i] < 0) and ema50_down_aligned[i] and vol_spike_1d_aligned[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bull Power <= 0 or 1d EMA50 turns down
            if (bull_power_aligned[i] <= 0) or (~ema50_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bear Power >= 0 or 1d EMA50 turns up
            if (bear_power_aligned[i] >= 0) or (~ema50_down_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals