#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Elder Ray (13-period) with 1d trend filter and volume confirmation.
# Bull power = EMA13(high) - EMA13(close), Bear power = EMA13(low) - EMA13(close).
# Long when bull power > 0 and bear power < 0 in uptrend, short when bear power < 0 and bull power > 0 in downtrend.
# Uses 1d EMA34 trend filter and volume spike confirmation to filter entries.
# Designed to work in both bull and bear markets by following the 1d trend direction.
# Target: 25-40 trades/year per symbol to avoid excessive fee drag.
name = "4h_ElderRay_13_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d trend filter: 34-period EMA on close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Elder Ray components (13-period EMA)
    ema13_high = pd.Series(high).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_low = pd.Series(low).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_close = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = ema13_high - ema13_close
    bear_power = ema13_low - ema13_close
    
    # 4h volume average for spike detection
    vol_ema_4h = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = np.where(vol_ema_4h > 0, volume / vol_ema_4h, 1.0) > 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup for EMA calculation
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long condition: bull power positive, bear power negative, in uptrend with volume spike
            long_condition = (bull_power[i] > 0) and (bear_power[i] < 0) and vol_spike[i] and uptrend
            # Short condition: bear power negative, bull power positive, in downtrend with volume spike
            short_condition = (bear_power[i] < 0) and (bull_power[i] > 0) and vol_spike[i] and downtrend
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: bear power turns positive (bullish momentum fading) or trend turns down
            if (bear_power[i] >= 0) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: bull power turns negative (bearish momentum fading) or trend turns up
            if (bull_power[i] <= 0) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals