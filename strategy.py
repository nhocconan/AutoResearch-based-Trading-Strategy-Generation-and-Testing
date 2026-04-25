#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike
Hypothesis: 4h Camarilla pivot R1/S1 breakout with 1d EMA34 trend filter and volume confirmation (>1.5x 20-bar avg). 
Enters long when price breaks above R1 in 1d uptrend with volume spike, short when price breaks below S1 in 1d downtrend with volume spike. 
Exits on opposite Camarilla level touch (S1 for longs exit, R1 for shorts exit) or trend reversal. 
Designed for 4h timeframe with ~20-50 trades/year, works in bull/bear by following 1d trend filter and momentum confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla pivot levels (based on previous 1d bar)
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    rng_1d = high_1d - low_1d
    camarilla_R1_1d = close_1d + (1.1 * rng_1d / 12)
    camarilla_S1_1d = close_1d - (1.1 * rng_1d / 12)
    
    # Align Camarilla levels to 4h timeframe (wait for completed 1d bar)
    camarilla_R1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1_1d)
    camarilla_S1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need at least 1 bar of previous data and EMA34 warmup
    start_idx = max(34, 20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_R1_1d_aligned[i]) or 
            np.isnan(camarilla_S1_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 in 1d uptrend with volume confirmation
            long_setup = (close[i] > camarilla_R1_1d_aligned[i]) and (close[i] > ema_34_1d_aligned[i]) and volume_spike[i]
            # Short: price breaks below S1 in 1d downtrend with volume confirmation
            short_setup = (close[i] < camarilla_S1_1d_aligned[i]) and (close[i] < ema_34_1d_aligned[i]) and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.30
                position = 1
            elif short_setup:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Exit: price touches S1 (opposite level) OR trend turns down
            if (close[i] < camarilla_S1_1d_aligned[i]) or (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit: price touches R1 (opposite level) OR trend turns up
            if (close[i] > camarilla_R1_1d_aligned[i]) or (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0