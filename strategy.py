#!/usr/bin/env python3
"""
12h Camarilla Pivot Breakout with Volume Spike and 1D EMA Trend Filter
Long: Price breaks above R1 + volume > 2.0x 12h volume MA + price > 1D EMA100
Short: Price breaks below S1 + volume > 2.0x 12h volume MA + price < 1D EMA100
Exit: Opposite break of S1 (long) or R1 (short)
Uses 1D Camarilla levels and 1D EMA100 to filter breakouts in low-volatility regimes
Target: 20-30 trades/year per symbol
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
    
    # Get 1D data for Camarilla pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for previous day
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Shift to get previous day's levels
    camarilla_r1_prev = camarilla_r1[1:]  # Shift forward to align with next day
    camarilla_s1_prev = camarilla_s1[1:]
    # Prepend NaN for first day
    camarilla_r1_prev = np.concatenate([np.array([np.nan]), camarilla_r1_prev])
    camarilla_s1_prev = np.concatenate([np.array([np.nan]), camarilla_s1_prev])
    
    # 1D EMA100 for trend filter
    ema_100_1d = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Align all 1D indicators to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_prev)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_prev)
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # 12h volume moving average (20-period for confirmation)
    df_12h = get_htf_data(prices, '12h')
    volume_ma_20 = pd.Series(df_12h['volume']).rolling(window=20, min_periods=20).mean()
    volume_ma_20_12h = align_htf_to_ltf(prices, df_12h, volume_ma_20.values)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_100_1d_aligned[i]) or np.isnan(volume_ma_20_12h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20_12h[i]
        
        if position == 0:
            # Long: break above R1 + volume spike + 1D uptrend
            if price > camarilla_r1_aligned[i] and vol > 2.0 * vol_ma and price > ema_100_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below S1 + volume spike + 1D downtrend
            elif price < camarilla_s1_aligned[i] and vol > 2.0 * vol_ma and price < ema_100_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: break below S1
            if price < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above R1
            if price > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_VolumeSpike_1DEMA100"
timeframe = "12h"
leverage = 1.0