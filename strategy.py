#!/usr/bin/env python3
"""
12h_1d_camarilla_breakout_volume_trend_v2
Hypothesis: On 12h timeframe, use daily Camarilla levels (S3/R3, S4/R4) for breakout signals with volume confirmation and 1-week trend filter.
Avoids false breakouts in sideways markets by requiring 1w trend alignment.
Targets 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.
Works in bull/bear via trend filter and volatility-adjusted breakouts.
"""

name = "12h_1d_camarilla_breakout_volume_trend_v2"
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
    
    # Get daily data for Camarilla and weekly data for trend
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous day's range for Camarilla
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Camarilla levels (based on previous day)
    range_ = prev_high - prev_low
    r3 = prev_close + range_ * 1.1 / 2
    r4 = prev_close + range_ * 1.1
    s3 = prev_close - range_ * 1.1 / 2
    s4 = prev_close - range_ * 1.1
    
    # 1-week EMA for trend filter (21-period)
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # ATR for volatility filter (14-day ATR)
    tr1 = np.abs(np.subtract(high_1d, low_1d))
    tr2 = np.abs(np.subtract(high_1d, np.roll(close_1d, 1)))
    tr3 = np.abs(np.subtract(low_1d, np.roll(close_1d, 1)))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Volume confirmation: volume > 1.3x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: close breaks above R4 with volume, volatility filter, and above weekly EMA
        if (close[i] > r4_aligned[i] and vol_confirm[i] and 
            atr_aligned[i] > 0 and close[i] > ema_21_1w_aligned[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: close breaks below S4 with volume, volatility filter, and below weekly EMA
        elif (close[i] < s4_aligned[i] and vol_confirm[i] and 
              atr_aligned[i] > 0 and close[i] < ema_21_1w_aligned[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or close crosses back to opposite S3/R3
        elif position == 1 and close[i] < s3_aligned[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > r3_aligned[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals