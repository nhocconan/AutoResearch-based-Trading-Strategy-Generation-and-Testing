#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 level breakout with 1d EMA34 trend filter and volume spike.
# Long when price breaks above Camarilla R3 level AND price > 1d EMA34 with volume spike.
# Short when price breaks below Camarilla S3 level AND price < 1d EMA34 with volume spike.
# Uses 1d EMA34 trend filter to align with higher timeframe trend and avoid counter-trend trades.
# Volume spike filter ensures momentum confirmation. Designed for fewer trades (target: 20-30/year) to reduce fee drag.
# Works in both bull and bear markets by following the 1d trend direction.
name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d trend filter: 34-period EMA on close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar (H, L, C from prior day)
    # H = previous day high, L = previous day low, C = previous day close
    H = df_1d['high'].shift(1).values
    L = df_1d['low'].shift(1).values
    C = df_1d['close'].shift(1).values
    
    # Camarilla R3 and S3 levels
    # R3 = C + (H - L) * 1.1/2
    # S3 = C - (H - L) * 1.1/2
    R3 = C + (H - L) * 1.1 / 2
    S3 = C - (H - L) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # 4h volume average for spike detection
    vol_ema_4h = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = np.where(vol_ema_4h > 0, volume / vol_ema_4h, 1.0) > 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long condition: break above Camarilla R3 level, in uptrend with volume spike
            long_condition = (close[i] > R3_aligned[i]) and uptrend and vol_spike[i]
            # Short condition: break below Camarilla S3 level, in downtrend with volume spike
            short_condition = (close[i] < S3_aligned[i]) and downtrend and vol_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Camarilla S3 level or trend turns down
            if (close[i] < S3_aligned[i]) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Camarilla R3 level or trend turns up
            if (close[i] > R3_aligned[i]) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals