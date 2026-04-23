#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above Camarilla R3 AND 1d EMA34 > 1w EMA34 (bullish HTF alignment) AND volume > 1.5x 20-period average.
Short when price breaks below Camarilla S3 AND 1d EMA34 < 1w EMA34 (bearish HTF alignment) AND volume > 1.5x 20-period average.
Exit when price touches the opposite Camarilla level (S3 for longs, R3 for shorts).
Uses 1d and 1w HTF for trend alignment and regime filtering. Target: 50-150 total trades over 4 years (12-37/year).
Camarilla levels provide precise intraday support/resistance; HTF EMA alignment ensures we trade with the dominant trend.
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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1w EMA34 for higher timeframe trend alignment
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF EMAs to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla levels (R3, S3) from previous 12h bar
    # Camarilla: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    # Using previous bar to avoid look-ahead
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # avoid NaN on first bar
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    camarilla_range = prev_high - prev_low
    camarilla_r3 = prev_close + 1.1 * camarilla_range / 2
    camarilla_s3 = prev_close - 1.1 * camarilla_range / 2
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34)  # volume MA (20), EMA calculation (34)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_1d = ema_34_1d_aligned[i]
        ema_1w = ema_34_1w_aligned[i]
        r3 = camarilla_r3[i]
        s3 = camarilla_s3[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Break above Camarilla R3 AND bullish HTF alignment (1d EMA > 1w EMA) AND volume spike
            if price > r3 and ema_1d > ema_1w and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S3 AND bearish HTF alignment (1d EMA < 1w EMA) AND volume spike
            elif price < s3 and ema_1d < ema_1w and volume[i] > 1.5 * vol_ma_val:
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

name = "12H_Camarilla_R3S3_Breakout_1dEMA34_1wEMA34_TrendAlign_VolumeConfirmation_LevelExit"
timeframe = "12h"
leverage = 1.0