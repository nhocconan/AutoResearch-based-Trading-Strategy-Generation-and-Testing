#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: 12h breakout at Camarilla R1/S1 levels in direction of 1d EMA34 trend, confirmed by volume spike (>2x 20-bar MA). Exits via opposite Camarilla level (S1 for longs, R1 for shorts). Uses discrete position sizing (0.25) to minimize fee drag. Camarilla levels provide structure in ranging markets while trend filter ensures alignment with higher timeframe momentum. Designed for 12-30 trades/year to avoid fee drag.
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size
    
    # Warmup: max of calculations
    start_idx = max(20, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_34_val = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine 1d trend: bullish if price > EMA34, bearish if price < EMA34
        bullish_1d = close_val > ema_34_val
        bearish_1d = close_val < ema_34_val
        
        # Camarilla levels for R1 and S1 (based on previous bar's range)
        if i >= 1:
            # Use previous bar's high, low, close for Camarilla calculation
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            range_val = prev_high - prev_low
            
            # Camarilla R1 and S1 levels
            camarilla_r1 = prev_close + (range_val * 1.1 / 12)
            camarilla_s1 = prev_close - (range_val * 1.1 / 12)
        else:
            camarilla_r1 = high_val
            camarilla_s1 = low_val
        
        # Entry conditions
        long_entry = (close_val > camarilla_r1) and bullish_1d and vol_spike
        short_entry = (close_val < camarilla_s1) and bearish_1d and vol_spike
        
        # Exit conditions: opposite Camarilla level touch
        exit_long = close_val < camarilla_s1
        exit_short = close_val > camarilla_r1
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0