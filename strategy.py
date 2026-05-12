#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_DailyVolumeSpike
Hypothesis: Price breaking above/below Camarilla R1/S1 levels (from previous day) 
with 4h EMA trend filter and daily volume confirmation (2x average) captures 
strong trending moves while avoiding false breakouts. Works in bull/bear by 
following 4h trend direction. Uses 1h timeframe for entry timing only.
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_DailyVolumeSpike"
timeframe = "1h"
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from 1d data (R1, S1)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Shift by 1 to use previous day's data
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = np.nan
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    camarilla_upper = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 12
    camarilla_lower = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe
    camarilla_upper_aligned = align_htf_to_ltf(prices, df_1d, camarilla_upper)
    camarilla_lower_aligned = align_htf_to_ltf(prices, df_1d, camarilla_lower)
    
    # Get 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA34 trend filter
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Daily volume spike: >2x 20-period average
    vol_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = df_1d['volume'].values > (2.0 * vol_20_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA34 warmup
        if (np.isnan(camarilla_upper_aligned[i]) or np.isnan(camarilla_lower_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R1 + 4h EMA34 uptrend + daily volume spike
            if (close[i] > camarilla_upper_aligned[i] and 
                close[i] > ema_34_4h_aligned[i] and 
                volume_spike_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below Camarilla S1 + 4h EMA34 downtrend + daily volume spike
            elif (close[i] < camarilla_lower_aligned[i] and 
                  close[i] < ema_34_4h_aligned[i] and 
                  volume_spike_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Camarilla S1 (reversal level)
            if close[i] < camarilla_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price closes above Camarilla R1 (reversal level)
            if close[i] > camarilla_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals