#!/usr/bin/env python3
name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume_v2"
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
    
    # Get 1d data for Camarilla pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    ph = df_1d['high'].shift(1).values
    pl = df_1d['low'].shift(1).values
    pc = df_1d['close'].shift(1).values
    
    # Camarilla R3, S3 (key reversal levels)
    r3 = pc + (ph - pl) * 1.1 / 2
    s3 = pc - (ph - pl) * 1.1 / 2
    
    # Align to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)
    
    for i in range(start_idx, n):
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(vol_avg[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 + uptrend + volume
            if close[i] > r3_aligned[i] and close[i] > ema_34_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + downtrend + volume
            elif close[i] < s3_aligned[i] and close[i] < ema_34_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price crosses back through the pivot level
            if position == 1:
                if close[i] < pc[i]:  # Exit below previous close
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > pc[i]:  # Exit above previous close
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals