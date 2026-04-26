#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA34_Trend_VolumeConfirmation
Hypothesis: Camarilla R1/S1 breakouts with 12h EMA34 trend filter and volume confirmation (>1.5x 20-bar MA). R1/S1 are inner Camarilla levels that act as strong support/resistance. Uses 4h timeframe for optimal trade frequency (target: 25-40 trades/year). Works in bull/bear markets by following 12h trend while using Camarilla structure for precise entries. Volume confirmation reduces false breakouts. Designed to generate ~80-120 total trades over 4 years with high win rate and low fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Previous day's OHLC for Camarilla levels (using 1d for structure)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Calculate Camarilla levels: R1, S1 (inner strong levels)
    rng = high_1d - low_1d
    camarilla_r1 = close_1d_vals + (rng * 1.1 / 12)  # R1 level
    camarilla_s1 = close_1d_vals - (rng * 1.1 / 12)  # S1 level
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size (25% of capital)
    
    # Warmup: max of calculations (20 for vol, 34 for 12h EMA, 1 for camarilla)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        ema_34_val = ema_34_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine 12h trend: bullish if price > EMA34, bearish if price < EMA34
        bullish_12h = close_val > ema_34_val
        bearish_12h = close_val < ema_34_val
        
        # Entry conditions: breakout of Camarilla R1/S1 in trend direction with volume spike
        long_entry = (close_val > camarilla_r1_val) and bullish_12h and vol_spike
        short_entry = (close_val < camarilla_s1_val) and bearish_12h and vol_spike
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
            elif short_entry:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on mean reversion to mid-point or trend change
            mid_point = (camarilla_r1_val + camarilla_s1_val) / 2
            if close_val < mid_point or not bullish_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = base_size
        elif position == -1:
            # Short - exit on mean reversion to mid-point or trend change
            mid_point = (camarilla_r1_val + camarilla_s1_val) / 2
            if close_val > mid_point or not bearish_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA34_Trend_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0