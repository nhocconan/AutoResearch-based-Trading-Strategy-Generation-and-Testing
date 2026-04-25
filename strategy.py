#!/usr/bin/env python3
"""
4h_Donchian20_VolumeSpike_HTFTrend_EMAFilter
Hypothesis: 4h Donchian(20) breakouts confirmed by volume spike (>2x 20-bar avg) and 1d EMA50 trend filter. 
Enters long on upper band break with volume in 1d uptrend (close > EMA50), short on lower band break with volume in 1d downtrend (close < EMA50). 
Donchian channels provide objective breakout levels; volume confirms conviction; 1d EMA50 ensures alignment with higher timeframe trend. 
Designed for 4h timeframe with ~20-40 trades/year via strict breakout + volume + trend confluence, avoiding overtrading and fee drag.
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
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian(20) on 4h timeframe
    donchian_window = 20
    upper = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian(20) and EMA50
    start_idx = max(donchian_window, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: upper band break with volume spike and 1d uptrend
            long_setup = (close[i] > upper[i]) and volume_spike[i] and (close[i] > ema_50_1d_aligned[i])
            # Short: lower band break with volume spike and 1d downtrend
            short_setup = (close[i] < lower[i]) and volume_spike[i] and (close[i] < ema_50_1d_aligned[i])
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: lower band break OR trend turns down
            if (close[i] < lower[i]) or (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: upper band break OR trend turns up
            if (close[i] > upper[i]) or (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_VolumeSpike_HTFTrend_EMAFilter"
timeframe = "4h"
leverage = 1.0