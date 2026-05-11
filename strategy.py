#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1dTrend_VolumeSurge"
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
    
    # Daily OHLC for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3, S3 levels: H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    # Actually: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # Standard: R3 = close + (high-low)*1.1/2, S3 = close - (high-low)*1.1/2
    hl_range = high_1d - low_1d
    r3 = close_1d + hl_range * 1.1 / 2
    s3 = close_1d - hl_range * 1.1 / 2
    
    # Align to 12h timeframe
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike detection (20-period EMA)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio = np.where(vol_ema20 > 0, volume / vol_ema20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(ema34_12h[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Breakout conditions with volume surge
        bullish_breakout = close[i] > r3_12h[i] and vol_ratio[i] > 2.0
        bearish_breakout = close[i] < s3_12h[i] and vol_ratio[i] > 2.0
        
        # Trend filter: price relative to daily EMA34
        uptrend = close[i] > ema34_12h[i]
        downtrend = close[i] < ema34_12h[i]
        
        if position == 0:
            # Long: bullish breakout + uptrend + volume surge
            if bullish_breakout and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakout + downtrend + volume surge
            elif bearish_breakout and downtrend:
                signals[i] = -0.25
                position = -1
        else:
            # Trailing exit: reverse signal or trend change
            if position == 1:
                if close[i] < s3_12h[i] or close[i] < ema34_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if close[i] > r3_12h[i] or close[i] > ema34_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals