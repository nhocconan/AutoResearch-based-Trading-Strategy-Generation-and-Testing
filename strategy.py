#!/usr/bin/env python3
"""
6h_aroon_oscillator_1d_trend_volume_v1
Hypothesis: Aroon oscillator from daily timeframe identifies strong trends 
(above +50 = strong uptrend, below -50 = strong downtrend). On 6h timeframe, 
we enter pullbacks in the direction of the daily trend with volume confirmation. 
This strategy works in both bull and bear markets by following the higher timeframe trend 
and avoids choppy markets by requiring strong Aroon readings (>|50|).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_aroon_oscillator_1d_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Aroon oscillator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Aroon oscillator (25-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    period = 25
    aroon_up = np.full(len(high_1d), np.nan)
    aroon_down = np.full(len(low_1d), np.nan)
    
    for i in range(period, len(high_1d)):
        # Periods since highest high
        highest_high_idx = np.argmax(high_1d[i-period:i+1]) + i - period
        periods_since_high = i - highest_high_idx
        aroon_up[i] = ((period - periods_since_high) / period) * 100
        
        # Periods since lowest low
        lowest_low_idx = np.argmin(low_1d[i-period:i+1]) + i - period
        periods_since_low = i - lowest_low_idx
        aroon_down[i] = ((period - periods_since_low) / period) * 100
    
    aroon_osc = aroon_up - aroon_down  # -100 to +100
    
    # Daily EMA20 for dynamic support/resistance
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False).mean().values
    
    # Aroon EMA for smoothing
    aroon_osc_ema = pd.Series(aroon_osc).ewm(span=5, adjust=False).mean().values
    
    # Align daily indicators to 6h timeframe
    aroon_osc_6h = align_htf_to_ltf(prices, df_1d, aroon_osc_ema)
    ema20_6h = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # 20-period volume average on 6h
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(aroon_osc_6h[i]) or np.isnan(ema20_6h[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: Aroon turns negative (trend weakness) OR price breaks below EMA20
            if aroon_osc_6h[i] < 0 or close[i] < ema20_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: Aroon turns positive (trend weakness) OR price breaks above EMA20
            if aroon_osc_6h[i] > 0 or close[i] > ema20_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Strong uptrend (Aroon > +50): buy pullbacks to EMA20 with volume
            if (aroon_osc_6h[i] > 50 and 
                vol_confirm and 
                close[i] <= ema20_6h[i] * 1.005):  # Allow small overshoot
                position = 1
                signals[i] = 0.25
            # Strong downtrend (Aroon < -50): sell rallies to EMA20 with volume
            elif (aroon_osc_6h[i] < -50 and 
                  vol_confirm and 
                  close[i] >= ema20_6h[i] * 0.995):  # Allow small undershoot
                position = -1
                signals[i] = -0.25
    
    return signals