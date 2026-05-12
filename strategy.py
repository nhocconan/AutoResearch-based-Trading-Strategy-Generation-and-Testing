#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeFilter
Hypothesis: Price breaking above/below Camarilla R1/S1 levels (derived from 4h high-low-close) with 4h EMA trend filter and volume confirmation (1.5x average) captures strong trending moves while avoiding false breakouts. Uses 1h for entry timing and 4h for direction to reduce noise and overtrading. Works in bull/bear by following 4h trend direction.
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeFilter"
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
    
    # Get 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Camarilla levels from 4h data
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C = close, H = high, L = low of previous 4h bar
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Shift by 1 to use previous 4h bar's data
    prev_close_4h = np.roll(close_4h, 1)
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h[0] = np.nan
    prev_high_4h[0] = np.nan
    prev_low_4h[0] = np.nan
    
    camarilla_upper = prev_close_4h + (prev_high_4h - prev_low_4h) * 1.1 / 12
    camarilla_lower = prev_close_4h - (prev_high_4h - prev_low_4h) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe
    camarilla_upper_aligned = align_htf_to_ltf(prices, df_4h, camarilla_upper)
    camarilla_lower_aligned = align_htf_to_ltf(prices, df_4h, camarilla_lower)
    
    # 4h EMA34 trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume spike: >1.5x 20-period average (1h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA34 warmup
        if (np.isnan(camarilla_upper_aligned[i]) or np.isnan(camarilla_lower_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(volume_spike[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R1 + 4h EMA34 uptrend + volume spike
            if (close[i] > camarilla_upper_aligned[i] and 
                close[i] > ema_34_4h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below Camarilla S1 + 4h EMA34 downtrend + volume spike
            elif (close[i] < camarilla_lower_aligned[i] and 
                  close[i] < ema_34_4h_aligned[i] and 
                  volume_spike[i]):
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