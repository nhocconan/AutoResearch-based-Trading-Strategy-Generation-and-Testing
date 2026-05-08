#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ChaikinOscillator_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Chaikin Oscillator (3,10) on 6d data
    # ADL = ((Close - Low) - (High - Close)) / (High - Low) * Volume
    # Handle division by zero
    hl_range = high - low
    hl_range = np.where(hl_range == 0, 1, hl_range)  # avoid div by zero
    adl = ((close - low) - (high - close)) / hl_range * volume
    # Cumulative ADL
    adl_cum = np.nancumsum(adl)
    # Chaikin Oscillator = EMA(3) - EMA(10) of ADL
    ema3_adl = pd.Series(adl_cum).ewm(span=3, adjust=False, min_periods=3).mean().values
    ema10_adl = pd.Series(adl_cum).ewm(span=10, adjust=False, min_periods=10).mean().values
    chaikin = ema3_adl - ema10_adl
    
    # Volume spike: current volume > 1.8x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(chaikin[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Chaikin > 0 (buying pressure) + 1d uptrend + volume spike
            long_cond = (chaikin[i] > 0 and 
                        ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and
                        volume_spike[i])
            
            # Short: Chaikin < 0 (selling pressure) + 1d downtrend + volume spike
            short_cond = (chaikin[i] < 0 and 
                         ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Chaikin turns negative (selling pressure)
            if chaikin[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Chaikin turns positive (buying pressure)
            if chaikin[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals