#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate previous day's Camarilla levels
    # Use previous day's OHLC (shifted by 1 to avoid look-ahead)
    ph = np.concatenate([[np.nan], high_1d[:-1]])  # previous day high
    pl = np.concatenate([[np.nan], low_1d[:-1]])   # previous day low
    pc = np.concatenate([[np.nan], close_1d[:-1]]) # previous day close
    
    # Camarilla levels: R1, S1, R3, S3
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    # R3 = C + (H-L)*1.1/4
    # S3 = C - (H-L)*1.1/4
    hl_range = ph - pl
    r1 = pc + hl_range * 1.1 / 12
    s1 = pc - hl_range * 1.1 / 12
    r3 = pc + hl_range * 1.1 / 4
    s3 = pc - hl_range * 1.1 / 4
    
    # EMA34 on 1d close for trend filter
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume spike (20-period average on 4h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    # Align all 1d indicators to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)  # EMA34 and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema34_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close breaks above R1 with volume spike and price > EMA34 (uptrend)
            if close[i] > r1_aligned[i] and vol_spike[i] and close[i] > ema34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S1 with volume spike and price < EMA34 (downtrend)
            elif close[i] < s1_aligned[i] and vol_spike[i] and close[i] < ema34_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close breaks below S1 (mean reversion)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close breaks above R1 (mean reversion)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals