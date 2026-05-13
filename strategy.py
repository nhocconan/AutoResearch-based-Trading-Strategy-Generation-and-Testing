#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
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
    
    # 1d Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ph = df_1d['high'].values
    pl = df_1d['low'].values
    pc = df_1d['close'].values
    
    R4 = pc + (1.1/2) * (ph - pl)
    R3 = pc + (1.1/4) * (ph - pl)
    R2 = pc + (1.1/6) * (ph - pl)
    R1 = pc + (1.1/12) * (ph - pl)
    S1 = pc - (1.1/12) * (ph - pl)
    S2 = pc - (1.1/6) * (ph - pl)
    S3 = pc - (1.1/4) * (ph - pl)
    S4 = pc - (1.1/2) * (ph - pl)
    
    R1_1d = align_htf_to_ltf(prices, df_1d, R1)
    S1_1d = align_htf_to_ltf(prices, df_1d, S1)
    
    # 1d EMA34 trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 4h volume filter: volume > 1.5 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        if (np.isnan(R1_1d[i]) or np.isnan(S1_1d[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        price_above_R1 = close[i] > R1_1d[i]
        price_below_S1 = close[i] < S1_1d[i]
        price_above_ema = close[i] > ema34_1d_aligned[i]
        price_below_ema = close[i] < ema34_1d_aligned[i]
        
        if position == 0:
            # LONG: break above R1 + uptrend + volume
            if price_above_R1 and price_above_ema and vol_filter:
                signals[i] = 0.25
                position = 1
            # SHORT: break below S1 + downtrend + volume
            elif price_below_S1 and price_below_ema and vol_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: close below S1 or trend change
            if price_below_S1 or price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: close above R1 or trend change
            if price_above_R1 or price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals