#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d Supertrend filter and volume confirmation.
Long when Bull Power > 0, Bear Power < 0, price > 13 EMA, and 1d Supertrend uptrend with volume spike.
Short when Bear Power < 0, Bull Power < 0, price < 13 EMA, and 1d Supertrend downtrend with volume spike.
Exit when Elder Ray signals weaken or Supertrend flips.
Designed for low trade frequency (15-35/year) to minimize fee flood.
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
    
    # Load daily data for Supertrend filter - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate Elder Ray (13-period EMA)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate 1d Supertrend (10, 3.0)
    high_d = pd.Series(df_daily['high'].values)
    low_d = pd.Series(df_daily['low'].values)
    close_d = pd.Series(df_daily['close'].values)
    
    # True Range
    tr1 = high_d - low_d
    tr2 = abs(high_d - close_d.shift(1))
    tr3 = abs(low_d - close_d.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_period = 10
    atr_d = tr.rolling(window=atr_period, min_periods=atr_period).mean()
    
    # Basic Upper and Lower Bands
    basic_ub = (high_d + low_d) / 2 + 3.0 * atr_d
    basic_lb = (high_d + low_d) / 2 - 3.0 * atr_d
    
    # Final Upper and Lower Bands
    final_ub = np.zeros(len(df_daily))
    final_lb = np.zeros(len(df_daily))
    
    for i in range(len(df_daily)):
        if i == 0:
            final_ub[i] = basic_ub[i]
            final_lb[i] = basic_lb[i]
        else:
            if basic_ub[i] < final_ub[i-1] or close_d[i-1] > final_ub[i-1]:
                final_ub[i] = basic_ub[i]
            else:
                final_ub[i] = final_ub[i-1]
                
            if basic_lb[i] > final_lb[i-1] or close_d[i-1] < final_lb[i-1]:
                final_lb[i] = basic_lb[i]
            else:
                final_lb[i] = final_lb[i-1]
    
    # Supertrend direction
    supertrend = np.zeros(len(df_daily))
    for i in range(len(df_daily)):
        if i == 0:
            supertrend[i] = 1 if close_d[i] <= final_ub[i] else -1
        else:
            if supertrend[i-1] == -1 and close_d[i] > final_ub[i]:
                supertrend[i] = 1
            elif supertrend[i-1] == 1 and close_d[i] < final_lb[i]:
                supertrend[i] = -1
            else:
                supertrend[i] = supertrend[i-1]
    
    # Align Supertrend to 6h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_daily, supertrend)
    
    # Calculate 6h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):
        # Skip if data not ready
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(supertrend_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, price > EMA13, Supertrend uptrend, volume spike
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                close[i] > ema13[i] and supertrend_aligned[i] == 1 and
                volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0, Bull Power < 0, price < EMA13, Supertrend downtrend, volume spike
            elif (bear_power[i] < 0 and bull_power[i] < 0 and 
                  close[i] < ema13[i] and supertrend_aligned[i] == -1 and
                  volume[i] > 2.0 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Elder Ray weakens OR Supertrend flips down
                if bull_power[i] <= 0 or bear_power[i] >= 0 or supertrend_aligned[i] == -1:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Elder Ray weakens OR Supertrend flips up
                if bear_power[i] >= 0 or bull_power[i] >= 0 or supertrend_aligned[i] == 1:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_1dSupertrend_Volume"
timeframe = "6h"
leverage = 1.0
#%%