#!/usr/bin/env python3
"""
6h Camarilla R3/S3 Breakout with 1d Trend Filter and Volume Spike
Hypothesis: Camarilla pivot levels provide high-probability breakout levels.
Breakouts above R3 or below S3 on 6h with 1d EMA trend alignment and volume spikes
capture strong trending moves while avoiding false breakouts in ranging markets.
Designed for low trade frequency (<30/year) to minimize fee drag in bear/bull markets.
"""
name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

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
    
    # === 1d EMA34 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Camarilla levels from previous 1d ===
    # Calculate daily high/low/close from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla multipliers
    R3 = close_1d + (high_1d - low_1d) * 1.1000
    S3 = close_1d - (high_1d - low_1d) * 1.1000
    R4 = close_1d + (high_1d - low_1d) * 1.5000
    S4 = close_1d - (high_1d - low_1d) * 1.5000
    
    # Align Camarilla levels to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # === Volume Spike (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)  # Require strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R3 with 1d uptrend and volume spike
            if (close[i] > R3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and  # 1d uptrend filter
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 with 1d downtrend and volume spike
            elif (close[i] < S3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and  # 1d downtrend filter
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price below S3 (reversal) OR 1d trend turns down
            if close[i] < S3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above R3 OR 1d trend turns up
            if close[i] > R3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals