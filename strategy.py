#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
Hypothesis: Camarilla pivot R1/S1 breakout with 1d EMA34 trend filter and volume spike on 4h timeframe.
Camarilla levels provide intraday support/resistance. Breakout of R1 (resistance 1) or S1 (support 1)
with volume confirmation indicates institutional interest. EMA34 filter ensures trend alignment.
Designed for 20-50 trades over 4 years. Works in bull via R1 breakouts above EMA34, bear via S1 breakdowns below EMA34.
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
    
    # Calculate 1d Camarilla pivot levels (R1, S1) from previous day
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla R1 and S1 levels
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate volume spike (volume > 1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    # Align all 1d indicators to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34)
    vol_spike_4h = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA and volume calculations
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(ema_34_4h[i]) or np.isnan(vol_spike_4h[i])):
            signals[i] = 0.0
            continue
        
        r1_val = r1_4h[i]
        s1_val = s1_4h[i]
        ema_34_val = ema_34_4h[i]
        vol_spike_val = vol_spike_4h[i]
        
        if position == 0:
            # Long: price breaks above R1 AND above EMA34 AND volume spike
            if close[i] > r1_val and close[i] > ema_34_val and vol_spike_val:
                signals[i] = size
                position = 1
            # Short: price breaks below S1 AND below EMA34 AND volume spike
            elif close[i] < s1_val and close[i] < ema_34_val and vol_spike_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price drops below S1 OR volume drops (optional)
            if close[i] < s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price rises above R1 OR volume drops (optional)
            if close[i] > r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0