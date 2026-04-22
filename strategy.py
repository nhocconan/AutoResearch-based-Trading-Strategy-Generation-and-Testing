#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1-week EMA13 trend filter and volume confirmation.
# Elder Ray measures bullish/bearish power by comparing high/low to EMA, capturing momentum exhaustion.
# Weekly EMA13 provides strong trend filter in both bull and bear markets.
# Volume spike (>2x 20-period average) confirms breakout strength.
# Designed for low trade frequency (~15-35/year) to minimize fee decay while capturing strong trends.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for EMA13 trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 13-period EMA on 1w close for trend filter
    ema_13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Align 1w EMA to 6h timeframe (waits for 1w bar to close)
    ema_13_aligned = align_htf_to_ltf(prices, df_1w, ema_13_1w)
    
    # Calculate 13-period EMA on 6h close for Elder Ray
    close_6h = prices['close'].values
    ema_13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    bull_power = high_6h - ema_13_6h  # Bull Power = High - EMA13
    bear_power = low_6h - ema_13_6h   # Bear Power = Low - EMA13
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_13_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bull = bull_power[i]
        bear = bear_power[i]
        ema_13_val = ema_13_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter: current volume > 2.0 * 20-period average (strict filter for low frequency)
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: Bull Power > 0 (bullish momentum) + above weekly EMA + volume spike
            if bull > 0 and ema_13_val > 0 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0 (bearish momentum) + below weekly EMA + volume spike
            elif bear < 0 and ema_13_val < 0 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Bull Power turns negative or weekly trend breaks
                if bull <= 0 or ema_13_val <= 0:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Bear Power turns positive or weekly trend breaks
                if bear >= 0 or ema_13_val >= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_1wEMA13_Volume"
timeframe = "6h"
leverage = 1.0