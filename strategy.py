#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation.
Long when price breaks above Camarilla R3 AND 1w EMA34 trending up AND volume > 1.5x 20-period average.
Short when price breaks below Camarilla S3 AND 1w EMA34 trending down AND volume > 1.5x 20-period average.
Exit when price touches opposite Camarilla level (S3 for longs, R3 for shorts).
Uses 1w HTF for EMA trend (avoids whipsaws in ranging markets). Target: 50-150 total trades over 4 years (12-37/year).
Camarilla levels provide precise intraday pivot points; EMA34 filter ensures we trade with the higher timeframe trend.
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
    
    # Calculate 1w EMA34 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate Camarilla levels from previous day (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Camarilla calculation: based on previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shifted by 1)
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    
    # Camarilla levels
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34 + 33, 1)  # volume MA (20), EMA calculation (34+33), 1d data (1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_1w_aligned[i]
        r3 = R3_aligned[i]
        s3 = S3_aligned[i]
        vol_ma_val = vol_ma[i]
        
        # Calculate EMA slope for trend direction (using previous value)
        if i > 0 and not np.isnan(ema_1w_aligned[i-1]):
            ema_slope = ema_val - ema_1w_aligned[i-1]
            ema_trending_up = ema_slope > 0
            ema_trending_down = ema_slope < 0
        else:
            ema_trending_up = False
            ema_trending_down = False
        
        if position == 0:
            # Long: Break above Camarilla R3 AND EMA trending up AND volume spike
            if price > r3 and ema_trending_up and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S3 AND EMA trending down AND volume spike
            elif price < s3 and ema_trending_down and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit when price touches opposite Camarilla level
            if position == 1 and price < s3:  # Long exit at Camarilla S3
                exit_signal = True
            elif position == -1 and price > r3:  # Short exit at Camarilla R3
                exit_signal = True
            else:
                exit_signal = False
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R3S3_Breakout_1wEMA34_Trend_VolumeConfirmation_LevelExit"
timeframe = "12h"
leverage = 1.0