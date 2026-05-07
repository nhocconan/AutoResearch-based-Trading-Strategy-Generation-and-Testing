#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray with volume confirmation and 1d trend filter.
# Uses Williams Alligator (Jaw/Teeth/Lips) for trend direction, Elder Ray for bull/bear power,
# and volume spike for confirmation. Designed to work in both bull and bear markets by following
# the 1d trend direction and filtering with Elder Ray. Target: 12-37 trades/year per symbol.
name = "12h_WilliamsAlligator_ElderRay_Volume_1dTrend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for trend filter and Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # 1d trend filter: 13-period EMA on close
    ema_13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = df_1d['high'].values - ema_13_1d
    bear_power = df_1d['low'].values - ema_13_1d
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs on median price
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    # Align Alligator components
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # 12h volume average for spike detection
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = np.where(vol_ema > 0, volume / vol_ema, 1.0) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams Alligator signals: Lips > Teeth > Jaw = uptrend, reverse = downtrend
        alligator_long = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        alligator_short = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
        
        # Elder Ray: Bull Power > 0 and Bear Power < 0 for strong trend
        elder_long = bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0
        elder_short = bull_power_aligned[i] < 0 and bear_power_aligned[i] > 0
        
        # 1d trend filter
        uptrend = close[i] > ema_13_1d_aligned[i]
        downtrend = close[i] < ema_13_1d_aligned[i]
        
        if position == 0:
            # Long: Alligator uptrend + Elder Ray bull power + 1d uptrend + volume spike
            long_condition = alligator_long and elder_long and uptrend and vol_spike[i]
            # Short: Alligator downtrend + Elder Ray bear power + 1d downtrend + volume spike
            short_condition = alligator_short and elder_short and downtrend and vol_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Alligator reverses or Elder Ray weakens or 1d trend turns down
            if not (alligator_long and elder_long and uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Alligator reverses or Elder Ray weakens or 1d trend turns up
            if not (alligator_short and elder_short and downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals