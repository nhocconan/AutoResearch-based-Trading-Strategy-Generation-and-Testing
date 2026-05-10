#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot breakouts (R3/S3) on 12h with 1d trend filter and volume confirmation.
# Works in bull markets via breakouts above R3 in uptrend, and in bear via breakdowns below S3 in downtrend.
# Volume filter ensures breakouts have conviction. Target: 15-25 trades/year to minimize fee drag.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate typical price for pivot
    typical_price = (high + low + close) / 3.0
    
    # Daily pivot calculation (using prior day's data)
    # We'll use rolling window to get previous day's OHLC
    # Since we're on 12h timeframe, we need 2 bars for previous day
    prev_day_high = pd.Series(high).rolling(window=2, min_periods=2).max().shift(1).values
    prev_day_low = pd.Series(low).rolling(window=2, min_periods=2).min().shift(1).values
    prev_day_close = pd.Series(close).rolling(window=2, min_periods=2).last().shift(1).values
    
    # Calculate pivot point
    pivot = (prev_day_high + prev_day_low + prev_day_close) / 3.0
    
    # Calculate Camarilla levels
    range_val = prev_day_high - prev_day_low
    r3 = pivot + (range_val * 1.1 / 2.0)  # R3 = pivot + 1.1*(H-L)/2
    s3 = pivot - (range_val * 1.1 / 2.0)  # S3 = pivot - 1.1*(H-L)/2
    
    # 1d trend filter: EMA(34) on daily closes
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(pivot) or np.isnan(r3) or np.isnan(s3) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: close above R3 AND price above 1d EMA34 (uptrend) AND volume confirmation
            if close[i] > r3 and close[i] > ema_34_1d_aligned[i] and volume[i] > vol_threshold[i]:
                signals[i] = 0.25
                position = 1
            # Short: close below S3 AND price below 1d EMA34 (downtrend) AND volume confirmation
            elif close[i] < s3 and close[i] < ema_34_1d_aligned[i] and volume[i] > vol_threshold[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: close below pivot OR trend bias lost
            if close[i] < pivot or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: close above pivot OR trend bias lost
            if close[i] > pivot or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals