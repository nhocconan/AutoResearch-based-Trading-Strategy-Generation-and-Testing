#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1-week trend filter and volume confirmation.
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13).
# Long when Bull Power > 0 and increasing, Bear Power < 0, in 1-week uptrend with volume spike.
# Short when Bear Power < 0 and decreasing, Bull Power > 0, in 1-week downtrend with volume spike.
# Uses 1-week EMA40 trend filter to avoid counter-trend trades in both bull and bear markets.
# Volume spike filter ensures momentum confirmation. Target: 15-30 trades/year for low fee drag.
name = "6h_ElderRay_1wEMA40_Volume"
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
    
    # Load 1w data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w trend filter: 40-period EMA on close
    ema_40_1w = pd.Series(df_1w['close']).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    # Elder Ray: EMA13 on close for 6h timeframe
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema_13
    # Bear Power = Low - EMA13
    bear_power = low - ema_13
    
    # 6h volume average for spike detection
    vol_ema_6h = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = np.where(vol_ema_6h > 0, volume / vol_ema_6h, 1.0) > 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Sufficient warmup for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(ema_40_1w_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1w EMA40
        uptrend = close[i] > ema_40_1w_aligned[i]
        downtrend = close[i] < ema_40_1w_aligned[i]
        
        # Elder Ray momentum: rising/falling power
        bull_rising = bull_power[i] > bull_power[i-1]
        bear_falling = bear_power[i] < bear_power[i-1]
        
        if position == 0:
            # Long condition: Bull Power > 0 and rising, Bear Power < 0, in uptrend with volume spike
            long_condition = (bull_power[i] > 0) and bull_rising and (bear_power[i] < 0) and uptrend and vol_spike[i]
            # Short condition: Bear Power < 0 and falling, Bull Power > 0, in downtrend with volume spike
            short_condition = (bear_power[i] < 0) and bear_falling and (bull_power[i] > 0) and downtrend and vol_spike[i]
            
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