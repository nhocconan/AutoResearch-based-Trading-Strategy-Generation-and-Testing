#!/usr/bin/env python3
"""
4h Camarilla Pivot Breakout with Volume Spike and 12h EMA Trend Filter
Strategy: Enter long when price breaks above R1 with volume spike and 12h EMA34 > 0,
          short when price breaks below S1 with volume spike and 12h EMA34 < 0.
          Uses daily Camarilla levels for breakout levels, 12h EMA34 for trend filter,
          and volume spike for confirmation. Designed for low trade frequency with clear breakout edge.
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
    
    # Get daily data for Camarilla pivot levels (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily close for Camarilla formula
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Calculate Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    daily_range = daily_high - daily_low
    camarilla_r1 = daily_close + daily_range * 1.1 / 12
    camarilla_s1 = daily_close - daily_range * 1.1 / 12
    
    # Get 12h data for EMA34 trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily Camarilla levels and 12h EMA to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_34_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_level = camarilla_r1_aligned[i]
        s1_level = camarilla_s1_aligned[i]
        ema_34 = ema_34_12h_aligned[i]
        
        if position == 0:
            # Long: break above R1 with volume spike and above 12h EMA34
            if (price > r1_level and volume_spike[i] and ema_34 > 0):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume spike and below 12h EMA34
            elif (price < s1_level and volume_spike[i] and ema_34 < 0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price breaks below S1 or below 12h EMA34
            if price < s1_level or ema_34 < 0:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price breaks above R1 or above 12h EMA34
            if price > r1_level or ema_34 > 0:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_VolumeSpike_12hEMA34"
timeframe = "4h"
leverage = 1.0