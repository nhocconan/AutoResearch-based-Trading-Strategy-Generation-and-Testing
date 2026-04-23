#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 Breakout with 1d EMA34 Trend Filter and Volume Spike
- Camarilla R1/S1 levels from 1d act as key support/resistance; breakout with volume indicates strong continuation
- 1d EMA34 defines the medium-term trend: only long when price > EMA34, short when price < EMA34
- Volume confirmation (> 1.5x 20-period average) reduces false breakouts
- Designed for 4h timeframe to capture swing trades with controlled frequency (target: 20-50 trades/year)
- Uses proven Camarilla structure with trend and volume filters to avoid overtrading
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Camarilla levels: R1/S1 = C ± (H-L)*1.1/12 (inner levels)
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align to 4h timeframe (use previous day's levels for breakout)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(1, 34, 20)  # need 1d pivots, 1d EMA34, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND above 1d EMA34 AND volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S1 AND below 1d EMA34 AND volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.30
                position = -1
        else:
            # Exit: price returns to Camarilla H3/L3 levels OR crosses 1d EMA34
            exit_signal = False
            # Calculate H3/L3 for exit
            camarilla_h3 = close_1d + (high_1d - low_1d) * 1.1 / 6
            camarilla_l3 = close_1d - (high_1d - low_1d) * 1.1 / 6
            camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
            camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
            
            if position == 1:
                # Exit long when price < H3 OR < 1d EMA34
                if close[i] < camarilla_h3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price > L3 OR > 1d EMA34
                if close[i] > camarilla_l3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0