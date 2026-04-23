#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA20 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R1 AND 4h EMA20 uptrend AND volume > 1.5x 20-period average.
Short when price breaks below Camarilla S1 AND 4h EMA20 downtrend AND volume > 1.5x 20-period average.
Exit when price retouches Camarilla pivot point (PP).
Uses discrete position sizing (0.20) to minimize fee churn. Targets 15-37 trades/year per symbol.
Camarilla levels provide intraday structure, 4h EMA20 ensures alignment with intermediate trend,
volume filters weak breakouts. Session filter (08-20 UTC) reduces noise trades.
Designed to work in both trending and ranging markets with strict entry conditions to avoid overtrading.
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
    
    # Calculate Camarilla levels from daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels (based on previous day's OHLC)
    camarilla_pp = (high_1d + low_1d + close_1d) / 3.0
    camarilla_r1 = camarilla_pp + (high_1d - low_1d) * 1.1 / 12.0
    camarilla_s1 = camarilla_pp - (high_1d - low_1d) * 1.1 / 12.0
    camarilla_r3 = camarilla_pp + (high_1d - low_1d) * 1.1 / 4.0
    camarilla_s3 = camarilla_pp - (high_1d - low_1d) * 1.1 / 4.0
    
    # Align Camarilla levels to 1h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Load 4h EMA20 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Volume average (20-period) on 1h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(ema20_4h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade between 08-20 UTC
        if hours[i] < 8 or hours[i] > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        pp = camarilla_pp_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        ema20 = ema20_4h_aligned[i]
        
        if position == 0:
            # Long: Price breaks above Camarilla R1 AND 4h EMA20 uptrend AND volume spike
            if (price > r1 and 
                close[i] > ema20 and  # Current close above EMA20 for uptrend
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below Camarilla S1 AND 4h EMA20 downtrend AND volume spike
            elif (price < s1 and 
                  close[i] < ema20 and  # Current close below EMA20 for downtrend
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retouches Camarilla pivot point
            if position == 1 and price <= pp:
                exit_signal = True
            elif position == -1 and price >= pp:
                exit_signal = True
            
            # Safety exit: Price breaks Camarilla R3/S3 (strong reversal)
            elif position == 1 and price >= r3:
                exit_signal = True
            elif position == -1 and price <= s3:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R1S1_4hEMA20_VolumeSpike"
timeframe = "1h"
leverage = 1.0