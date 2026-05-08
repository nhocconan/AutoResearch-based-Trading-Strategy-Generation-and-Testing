#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Chaikin_Oscillator_1dTrend_Volume"
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
    
    # 1d data for trend filter and Chaikin Oscillator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA10 and EMA3 for Chaikin Oscillator components
    close_1d_series = pd.Series(close_1d)
    ema3_1d = close_1d_series.ewm(span=3, adjust=False, min_periods=3).mean().values
    ema10_1d = close_1d_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Money Flow Multiplier and Volume for ADL calculation
    mfm = ((close_1d - low_1d) - (high_1d - close_1d)) / (high_1d - low_1d)
    mfm = np.where((high_1d - low_1d) == 0, 0, mfm)
    mfv = mfm * volume_1d
    
    # Accumulation/Distribution Line (ADL)
    adl = np.cumsum(mfv)
    
    # Chaikin Oscillator: EMA3(ADL) - EMA10(ADL)
    adl_series = pd.Series(adl)
    ema3_adl = adl_series.ewm(span=3, adjust=False, min_periods=3).mean().values
    ema10_adl = adl_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    chaikin_osc = ema3_adl - ema10_adl
    
    # Align Chaikin Oscillator to 12h timeframe
    chaikin_osc_aligned = align_htf_to_ltf(prices, df_1d, chaikin_osc)
    
    # 1d EMA34 for trend filter
    ema34_1d_series = pd.Series(close_1d)
    ema34_1d = ema34_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current volume > 1.8x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(chaikin_osc_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Chaikin Oscillator > 0 with 1d uptrend + volume spike
            long_cond = (chaikin_osc_aligned[i] > 0 and 
                        ema34_1d_aligned[i] > ema34_1d_aligned[i-1] and
                        volume_spike[i])
            
            # Short: Chaikin Oscillator < 0 with 1d downtrend + volume spike
            short_cond = (chaikin_osc_aligned[i] < 0 and 
                         ema34_1d_aligned[i] < ema34_1d_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Chaikin Oscillator crosses below 0
            if chaikin_osc_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Chaikin Oscillator crosses above 0
            if chaikin_osc_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals