#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_R1S1_Breakout_1dTrend_VolumeSpike
Hypothesis: On 12h timeframe, breakouts above R1 or below S1 (from 1d Camarilla pivot levels) with 1d trend filter and volume spike capture strong moves while avoiding whipsaws. The 1d EMA34 provides longer-term trend context, reducing false signals in choppy markets. Volume surge confirms breakout conviction. Designed for low trade frequency (12-37/year) to minimize fee drag on 12h timeframe.
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
    
    # Get 1d data for Camarilla pivot levels and trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    
    # Calculate Camarilla levels for each day
    # R1 = Close + 1.1*(High-Low)/12
    # S1 = Close - 1.1*(High-Low)/12
    camarilla_range = (high_1d - low_1d)
    r1_level = close_1d + (1.1 * camarilla_range) / 12
    s1_level = close_1d - (1.1 * camarilla_range) / 12
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_level)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_level)
    
    # Get 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: break above R1 with 1d uptrend and volume spike
            if price > r1 and price > ema_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with 1d downtrend and volume spike
            elif price < s1 and price < ema_trend and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price returns to S1 or breaks below 1d EMA
            if price < s1 or price < ema_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price returns to R1 or breaks above 1d EMA
            if price > r1 or price > ema_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_Pivot_R1S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0