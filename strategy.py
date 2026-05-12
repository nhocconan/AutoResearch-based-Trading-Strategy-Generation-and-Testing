#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 12h Camarilla pivot levels from previous 12h bar
    high_12h = pd.Series(high).rolling(window=2, min_periods=2).max().shift(1).values
    low_12h = pd.Series(low).rolling(window=2, min_periods=2).min().shift(1).values
    close_12h = pd.Series(close).rolling(window=2, min_periods=2).mean().shift(1).values
    R3 = close_12h + (high_12h - low_12h) * 1.1 / 4
    S3 = close_12h - (high_12h - low_12h) * 1.1 / 4
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # need enough data for EMA34 and Camarilla
    
    for i in range(start_idx, n):
        # Skip if 1d trend data not ready
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(R3[i]) or np.isnan(S3[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 AND volume confirmation AND 1d uptrend
            if (close[i] > R3[i] and 
                vol_ratio[i] > 1.5 and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.30
                position = 1
            # Short: Price breaks below S3 AND volume confirmation AND 1d downtrend
            elif (close[i] < S3[i] and 
                  vol_ratio[i] > 1.5 and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: Price breaks below S3 OR opposite signal with volume
            if (close[i] < S3[i] and vol_ratio[i] > 1.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: Price breaks above R3 OR opposite signal with volume
            if (close[i] > R3[i] and vol_ratio[i] > 1.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals