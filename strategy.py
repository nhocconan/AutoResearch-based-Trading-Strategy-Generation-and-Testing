#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA40 trend filter and volume spike.
# Bull Power = High - EMA13, Bear Power = Low - EMA13. EMA13 on 6h closes.
# Long when Bull Power > 0 AND price > 1d EMA40 AND volume spike.
# Short when Bear Power < 0 AND price < 1d EMA40 AND volume spike.
# Uses 1d EMA40 trend filter to align with higher timeframe trend and avoid counter-trend trades.
# Volume spike filter ensures momentum confirmation. Designed for 15-25 trades/year.
# Works in both bull and bear markets by following the 1d trend direction.
name = "6h_ElderRay_1dEMA40_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d trend filter: 40-period EMA on close
    ema_40_1d = pd.Series(df_1d['close']).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_40_1d)
    
    # 6h EMA13 for Elder Ray calculation
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13_6h
    bear_power = low - ema_13_6h
    
    # 6h volume average for spike detection
    vol_ema_6h = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = np.where(vol_ema_6h > 0, volume / vol_ema_6h, 1.0) > 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(ema_40_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA40
        uptrend = close[i] > ema_40_1d_aligned[i]
        downtrend = close[i] < ema_40_1d_aligned[i]
        
        if position == 0:
            # Long condition: Bull Power > 0, in uptrend with volume spike
            long_condition = (bull_power[i] > 0) and uptrend and vol_spike[i]
            # Short condition: Bear Power < 0, in downtrend with volume spike
            short_condition = (bear_power[i] < 0) and downtrend and vol_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bull Power <= 0 or trend turns down
            if (bull_power[i] <= 0) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bear Power >= 0 or trend turns up
            if (bear_power[i] >= 0) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals