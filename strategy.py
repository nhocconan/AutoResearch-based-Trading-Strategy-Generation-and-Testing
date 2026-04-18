#!/usr/bin/env python3
"""
4h_12h_ChaikinMoneyFlow_R1S1_Breakout_Volume
Hypothesis: Use Chaikin Money Flow (CMF) from 12h as institutional flow filter combined with 1d Camarilla R1/S1 breakout on 4h. CMF > 0 indicates buying pressure, CMF < 0 selling pressure. Only long when CMF > 0.05 and price breaks above R1; only short when CMF < -0.05 and price breaks below S1. Volume confirmation requires current volume > 1.8x 20-period average. Targets 20-35 trades/year by requiring alignment of institutional flow (CMF), price breakout beyond daily R1/S1, and high volume. Works in bull markets by following institutional buying into R1 breakouts, and in bear markets by following institutional selling into S1 breaks.
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
    
    # Get 1d data for Camarilla levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla R1 and S1
    rng_1d = high_1d - low_1d
    r1_1d = close_1d + rng_1d * 1.1 / 12
    s1_1d = close_1d - rng_1d * 1.1 / 12
    
    # Align levels to 4h timeframe (wait for bar close)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Get 12h data for Chaikin Money Flow (HTF)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Chaikin Money Flow (CMF) over 20 periods
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    mfm = np.zeros_like(close_12h)
    mfm_denom = high_12h - low_12h
    mfm_denom = np.where(mfm_denom == 0, 1, mfm_denom)  # avoid div by zero
    mfm = ((close_12h - low_12h) - (high_12h - close_12h)) / mfm_denom
    
    # Money Flow Volume = MFM * Volume
    mfv = mfm * volume_12h
    
    # CMF = 20-period sum of MFV / 20-period sum of Volume
    cmf = np.full_like(close_12h, np.nan)
    for i in range(20, len(close_12h)):
        mfv_sum = np.sum(mfv[i-20:i])
        vol_sum = np.sum(volume_12h[i-20:i])
        cmf[i] = mfv_sum / vol_sum if vol_sum != 0 else 0
    
    # Align CMF to 4h timeframe
    cmf_aligned = align_htf_to_ltf(prices, df_12h, cmf)
    
    # Volume confirmation: current volume > 1.8 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA and CMF
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(cmf_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above 1d R1, with volume, and CMF positive (buying pressure)
            if (close[i] > r1_1d_aligned[i] and vol_confirm[i] and 
                cmf_aligned[i] > 0.05):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below 1d S1, with volume, and CMF negative (selling pressure)
            elif (close[i] < s1_1d_aligned[i] and vol_confirm[i] and 
                  cmf_aligned[i] < -0.05):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns below R1 (failed breakout) or CMF turns negative
            if (close[i] < r1_1d_aligned[i] or 
                cmf_aligned[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above S1 (failed breakout) or CMF turns positive
            if (close[i] > s1_1d_aligned[i] or 
                cmf_aligned[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_ChaikinMoneyFlow_R1S1_Breakout_Volume"
timeframe = "4h"
leverage = 1.0