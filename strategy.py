#!/usr/bin/env python3
name = "1d_Camarilla_Pivot_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA34 for trend
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Camarilla levels from previous day
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    
    camarilla_r4 = np.zeros(n)
    camarilla_r3 = np.zeros(n)
    camarilla_r2 = np.zeros(n)
    camarilla_r1 = np.zeros(n)
    camarilla_s1 = np.zeros(n)
    camarilla_s2 = np.zeros(n)
    camarilla_s3 = np.zeros(n)
    camarilla_s4 = np.zeros(n)
    
    # Shift high/low/close by 1 to use previous day's values
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # First bar has no previous day
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    diff = prev_high - prev_low
    camarilla_r4 = prev_close + (diff * 1.1 / 2)
    camarilla_r3 = prev_close + (diff * 1.1 / 4)
    camarilla_r2 = prev_close + (diff * 1.1 / 6)
    camarilla_r1 = prev_close + (diff * 1.1 / 12)
    camarilla_s1 = prev_close - (diff * 1.1 / 12)
    camarilla_s2 = prev_close - (diff * 1.1 / 6)
    camarilla_s3 = prev_close - (diff * 1.1 / 4)
    camarilla_s4 = prev_close - (diff * 1.1 / 2)
    
    # Volume spike (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for calculations
    start_idx = 20
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: close breaks above R1 with volume spike and weekly uptrend
            if close[i] > camarilla_r1[i] and vol_spike[i] and close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: close breaks below S1 with volume spike and weekly downtrend
            elif close[i] < camarilla_s1[i] and vol_spike[i] and close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: close breaks below S1 (reversal signal)
            if close[i] < camarilla_s1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: close breaks above R1 (reversal signal)
            if close[i] > camarilla_r1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals