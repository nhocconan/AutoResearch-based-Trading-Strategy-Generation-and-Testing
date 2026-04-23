#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation.
Uses 4h/1d for signal direction (trend/volume) and 1h for precise entry timing.
Session filter (08-20 UTC) reduces noise. Target 15-37 trades/year per symbol.
Position size 0.20 balances edge with fee drag. Works in bull/bear via trend filter.
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
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d ATR(14) for volume spike threshold
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.maximum(high_1d[1:], low_1d[:-1]) - np.minimum(low_1d[1:], high_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 1h Camarilla levels (using prior day's range)
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # We'll use prior 1d high/low to compute today's levels
    high_1d_shifted = np.concatenate([[np.nan], high_1d[:-1]])
    low_1d_shifted = np.concatenate([[np.nan], low_1d[:-1]])
    close_1d_shifted = np.concatenate([[np.nan], close_1d[:-1]])
    
    rangep = high_1d_shifted - low_1d_shifted
    r3 = close_1d_shifted + rangep * 1.1 / 4
    s3 = close_1d_shifted - rangep * 1.1 / 4
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate 1h volume MA(20) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14)  # need EMA50_4h and ATR14_1d
    
    for i in range(start_idx, n):
        # Skip if not in session or data not ready
        if not in_session[i] or np.isnan(ema_50_4h_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or \
           np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: close > 4h EMA50 = uptrend, close < 4h EMA50 = downtrend
        trend_up = close[i] > ema_50_4h_aligned[i]
        trend_down = close[i] < ema_50_4h_aligned[i]
        
        # Volume filter: 1h volume > 2.0 x 1d ATR (volume spike)
        vol_filter = volume[i] > 2.0 * atr_14_1d_aligned[i]
        
        if position == 0:
            # Long: Break above R3 AND uptrend AND volume confirmation
            if close[i] > r3_aligned[i] and trend_up and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: Break below S3 AND downtrend AND volume confirmation
            elif close[i] < s3_aligned[i] and trend_down and vol_filter:
                signals[i] = -0.20
                position = -1
        else:
            # Exit: break of opposite Camarilla level (S3 for longs, R3 for shorts)
            exit_signal = False
            if position == 1:
                # Exit long on break below S3
                if close[i] < s3_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short on break above R3
                if close[i] > r3_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R3S3_Breakout_4hEMA50_Trend_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0