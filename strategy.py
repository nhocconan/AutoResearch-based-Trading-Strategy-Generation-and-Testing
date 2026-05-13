#!/usr/bin/env python3
name = "6h_Camarilla_R3S3_Breakout_12hTrend_VolumeSqueeze"
timeframe = "6h"
leverage = 1.0

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
    
    # Calculate 12h Camarilla pivot levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    # Daily high, low, close for Camarilla calculation
    ph = df_12h['high'].values
    pl = df_12h['low'].values
    pc = df_12h['close'].values
    
    # Camarilla levels: R3, R4, S3, S4
    r3 = pc + (ph - pl) * 1.1 / 4
    r4 = pc + (ph - pl) * 1.1 / 2
    s3 = pc - (ph - pl) * 1.1 / 4
    s4 = pc - (ph - pl) * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    # 12h trend filter: EMA(50)
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume squeeze: current volume < 0.5 x 20-period average (low volume breakout)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    volume_filter = volume < 0.5 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_squeeze = volume_filter[i]
        price_above_r3 = close[i] > r3_aligned[i]
        price_above_r4 = close[i] > r4_aligned[i]
        price_below_s3 = close[i] < s3_aligned[i]
        price_below_s4 = close[i] < s4_aligned[i]
        price_above_ema = close[i] > ema50_12h_aligned[i]
        price_below_ema = close[i] < ema50_12h_aligned[i]
        
        if position == 0:
            # LONG: breakout above R3 with volume squeeze and 12h uptrend
            if price_above_r3 and vol_squeeze and price_above_ema:
                signals[i] = 0.25
                position = 1
            # SHORT: breakdown below S3 with volume squeeze and 12h downtrend
            elif price_below_s3 and vol_squeeze and price_below_ema:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price below R4 or trend breaks
            if price_below_r4 or price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price above S4 or trend breaks
            if price_above_s4 or price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals