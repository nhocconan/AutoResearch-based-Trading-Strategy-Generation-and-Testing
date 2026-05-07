#!/usr/bin/env python3
name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume_Spike"
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # 4h EMA34 trend filter
    ema_34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 1h Camarilla pivot from previous 4h candle
    # Camarilla levels: H4 = C + 1.5*(H-L), L4 = C - 1.5*(H-L)
    # where H,L,C are from previous 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    camarilla_H4 = close_4h + 1.5 * (high_4h - low_4h)
    camarilla_L4 = close_4h - 1.5 * (high_4h - low_4h)
    
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_L4)
    
    # Volume filter: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need at least 20 bars for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_4h_aligned[i]) or np.isnan(camarilla_H4_aligned[i]) or np.isnan(camarilla_L4_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Camarilla H4 + above 4h EMA34 + volume spike
            if close[i] > camarilla_H4_aligned[i] and close[i] > ema_34_4h_aligned[i] and volume_ok[i]:
                signals[i] = 0.20
                position = 1
            # Short: break below Camarilla L4 + below 4h EMA34 + volume spike
            elif close[i] < camarilla_L4_aligned[i] and close[i] < ema_34_4h_aligned[i] and volume_ok[i]:
                signals[i] = -0.20
                position = -1
        elif position != 0:
            # Exit: price returns to Camarilla range or breaks in opposite direction
            if position == 1:
                if close[i] < camarilla_L4_aligned[i] or close[i] < ema_34_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if close[i] > camarilla_H4_aligned[i] or close[i] > ema_34_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals