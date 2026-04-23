#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above R3 AND 1w close > EMA50 AND volume > 1.5x 20-period average.
Short when price breaks below S3 AND 1w close < EMA50 AND volume > 1.5x 20-period average.
Exit when price touches opposite Camarilla level (S3 for longs, R3 for shorts).
Uses 1w HTF for trend strength to avoid whipsaws in ranging markets. Target: 50-150 total trades over 4 years (12-37/year).
Camarilla levels work well in crypto due to institutional reaction at these levels. 1w EMA filter ensures we only trade with the weekly trend.
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
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 12h Camarilla levels (based on previous bar's OHLC)
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    camarilla_r3 = close + (high - low) * 1.1 / 2
    camarilla_s3 = close - (high - low) * 1.1 / 2
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50 + 49)  # volume MA (20), EMA calculation (50+49)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_trend = ema_50_1w_aligned[i]
        r3 = camarilla_r3[i]
        s3 = camarilla_s3[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Break above R3 AND 1w close > EMA50 AND volume spike
            if price > r3 and close_1w[-1] > ema_50_1w[-1] if len(close_1w) > 0 else False and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 AND 1w close < EMA50 AND volume spike
            elif price < s3 and close_1w[-1] < ema_50_1w[-1] if len(close_1w) > 0 else False and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit when price touches opposite Camarilla level
            if position == 1 and price < s3:  # Long exit at S3
                exit_signal = True
            elif position == -1 and price > r3:  # Short exit at R3
                exit_signal = True
            else:
                exit_signal = False
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R3S3_Breakout_1wEMA50_Trend_VolumeConfirmation_LevelExit"
timeframe = "12h"
leverage = 1.0