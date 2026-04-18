#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_With_Volume_Spike_And_1dTrend
Hypothesis: Trade reversals at Camarilla pivot levels (H3/L3) from 1d timeframe with volume spike confirmation and 1d EMA trend filter. Designed for low trade frequency to minimize fee drag while capturing mean-reversion bounces at key intraday levels in both bull and bear markets.
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
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    ph = df_1d['high'].shift(1).values  # previous day high
    pl = df_1d['low'].shift(1).values   # previous day low
    pc = df_1d['close'].shift(1).values # previous day close
    
    # Camarilla levels: H3/L3 (most important for reversals)
    camarilla_h3 = pc + (ph - pl) * 1.1 / 4
    camarilla_l3 = pc - (ph - pl) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (wait for daily close)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 1d EMA34 trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 40  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        vol_spike = volume_spike[i]
        ema_1d_val = ema_1d_aligned[i]
        
        if position == 0:
            # Long: price near L3 with volume spike and above 1d EMA34
            if price <= l3 * 1.005 and price >= l3 * 0.995 and vol_spike and price > ema_1d_val:
                signals[i] = 0.25
                position = 1
            # Short: price near H3 with volume spike and below 1d EMA34
            elif price >= h3 * 0.995 and price <= h3 * 1.005 and vol_spike and price < ema_1d_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price moves to H3 or below 1d EMA34
            if price >= h3 * 0.995 or price < ema_1d_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price moves to L3 or above 1d EMA34
            if price <= l3 * 1.005 or price > ema_1d_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_Pivot_With_Volume_Spike_And_1dTrend"
timeframe = "12h"
leverage = 1.0