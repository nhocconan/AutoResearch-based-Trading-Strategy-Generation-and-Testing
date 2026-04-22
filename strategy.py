#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1-week EMA13 trend filter and volume spike confirmation.
# Elder Ray measures bull power (high - EMA) and bear power (low - EMA) to identify trend strength.
# Long when bull power > 0 and rising, bear power < 0, with price above weekly EMA13 and volume spike.
# Short when bear power < 0 and falling, bull power > 0, with price below weekly EMA13 and volume spike.
# Works in bull markets via bull power strength and in bear markets via bear power strength.
# Weekly EMA13 filter ensures alignment with longer-term trend, reducing whipsaw.
# Volume spike (>2x 20-period average) confirms institutional participation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for EMA13 trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 13-period EMA on 1w close for trend filter
    ema_13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Align 1w EMA to 6h timeframe (waits for 1w bar to close)
    ema_13_aligned = align_htf_to_ltf(prices, df_1w, ema_13_1w)
    
    # Calculate 13-period EMA on 6h close for Elder Ray
    close = prices['close'].values
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high_1w - ema_13  # Using weekly high for strength
    bear_power = low_1w - ema_13   # Using weekly low for weakness
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1w, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1w, bear_power)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_13_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema_val = ema_13_aligned[i]
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: bull power positive AND rising, bear power negative, price above weekly EMA, volume spike
            if (i > 50 and 
                bull_val > 0 and 
                bull_val > bull_power_aligned[i-1] and  # rising bull power
                bear_val < 0 and 
                price > ema_val and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short conditions: bear power negative AND falling, bull power positive, price below weekly EMA, volume spike
            elif (i > 50 and 
                  bear_val < 0 and 
                  bear_val < bear_power_aligned[i-1] and  # falling bear power
                  bull_val > 0 and 
                  price < ema_val and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when bull power turns negative or price breaks below weekly EMA
                if bull_val <= 0 or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when bear power turns positive or price breaks above weekly EMA
                if bear_val >= 0 or price > ema_val:
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