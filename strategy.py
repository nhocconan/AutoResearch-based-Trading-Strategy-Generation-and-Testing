#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla Pivot R1/S1 Breakout with 1w EMA34 Trend Filter and Volume Spike
- Camarilla R1/S1 levels from 1d act as key support/resistance; breakout with volume indicates continuation
- 1w EMA34 defines the primary trend: only long when price > EMA34, short when price < EMA34
- Volume confirmation (> 2.0x 30-period average) reduces false breakouts
- Designed for 1d timeframe to capture swing trades with low frequency (~10-25 trades/year)
- Works in bull via long breakouts above R1 and in bear via short breakdowns below S1
- Uses discrete position sizing (0.25) to minimize fee churn
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
    
    # Calculate 1d Camarilla pivots (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R1/S1 = C ± (H-L)*1.1/2
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align to 1d timeframe (use previous day's levels for breakout)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: > 2.0x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(1, 34, 30)  # need 1d pivots, 1w EMA34, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND above 1w EMA34 AND volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND below 1w EMA34 AND volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to Camarilla H1/L1 levels OR crosses 1w EMA34
            exit_signal = False
            # Calculate H1/L1 for exit (inner levels)
            camarilla_h1 = close_1d + (high_1d - low_1d) * 1.1/4
            camarilla_l1 = close_1d - (high_1d - low_1d) * 1.1/4
            camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
            camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
            
            if position == 1:
                # Exit long when price < H1 OR < 1w EMA34
                if close[i] < camarilla_h1_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price > L1 OR > 1w EMA34
                if close[i] > camarilla_l1_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0