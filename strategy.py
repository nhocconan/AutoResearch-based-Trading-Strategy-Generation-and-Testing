#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "12h"
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
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 1w EMA(34) for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 12h Camarilla pivot levels from previous day (1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels: R3, R2, R1, PP, S1, S2, S3
    # Using previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point
    pp = (high_1d + low_1d + close_1d) / 3
    # Ranges
    range_ = high_1d - low_1d
    # Camarilla levels
    r3 = pp + (range_ * 1.1 / 2)
    r2 = pp + (range_ * 1.1 / 4)
    r1 = pp + (range_ * 1.1 / 6)
    s1 = pp - (range_ * 1.1 / 6)
    s2 = pp - (range_ * 1.1 / 4)
    s3 = pp - (range_ * 1.1 / 2)
    
    # Align to 12h timeframe (use previous day's levels)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    pp_12h = align_htf_to_ltf(prices, df_1d, pp)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    
    # 12h volume spike (24-period average)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Wait for volume MA and alignment
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or np.isnan(vol_ma_24[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R3 with volume and 1w uptrend
            vol_condition = volume[i] > vol_ma_24[i] * 2.0
            uptrend = ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]
            
            if close[i] > r3_12h[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 with volume and 1w downtrend
            elif close[i] < s3_12h[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below R1 or volume drops
            if close[i] < r1_12h[i] or volume[i] < vol_ma_24[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above S1 or volume drops
            if close[i] > s1_12h[i] or volume[i] < vol_ma_24[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA trend filter and volume confirmation
# - Camarilla R3/S3 levels act as strong support/resistance - breaks indicate institutional interest
# - 1w EMA(34) ensures alignment with higher timeframe trend (works in bull/bear markets)
# - Volume spike (2.0x average) confirms genuine breakout with participation
# - Exit at R1/S1 provides profit target in ranging markets
# - Position size 0.25 targets 20-40 trades/year, avoiding excessive fee drag
# - Works in both bull (buy R3 breaks in uptrend) and bear (sell S3 breaks in downtrend) markets