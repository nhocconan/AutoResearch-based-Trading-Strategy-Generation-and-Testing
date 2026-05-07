#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

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
    
    # Load 1d data ONCE for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (R1, S1) from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection (4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: break above R1 in 1d uptrend with volume
            if high[i] > camarilla_r1_aligned[i] and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 in 1d downtrend with volume
            elif low[i] < camarilla_s1_aligned[i] and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below S1 or trend change
            if low[i] < camarilla_s1_aligned[i] or ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above R1 or trend change
            if high[i] > camarilla_r1_aligned[i] or ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend and volume confirmation
# - Uses actual 1d Camarilla pivot levels (R1, S1) from previous day
# - Enters long when price breaks above R1 in 1d uptrend with volume spike
# - Enters short when price breaks below S1 in 1d downtrend with volume spike
# - Exits when price returns to opposite level or trend changes
# - Volume confirmation (1.5x average) reduces false breakouts
# - Position size 0.25 targets ~20-50 trades/year to avoid fee drag
# - Camarilla levels provide institutional support/resistance with clear breakout levels
# - 1d trend filter ensures alignment with higher timeframe direction
# - Works in both bull (R1 breakouts in uptrend) and bear (S1 breakdowns in downtrend)