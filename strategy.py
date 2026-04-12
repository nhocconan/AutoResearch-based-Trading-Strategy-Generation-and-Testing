#!/usr/bin/env python3
# 1d_1w_camarilla_breakout_volume_regime_v2
# Hypothesis: Daily Camarilla breakout with weekly trend filter and volume confirmation.
# Uses weekly EMA to filter long/short bias, reducing false signals in chop.
# Volume > 1.3x 20-day average confirms breakout strength.
# Target: 15-25 trades/year (60-100 total) to minimize fee drag.
# Works in bull/bear by aligning with weekly trend.

name = "1d_1w_camarilla_breakout_volume_regime_v2"
timeframe = "1d"
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
    
    # Get daily data for Camarilla
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Previous day's range for Camarilla
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close = np.roll(df_1d['close'].values, 1)
    
    range_ = prev_high - prev_low
    # Resistance levels
    r3 = prev_close + range_ * 1.1 / 2
    r4 = prev_close + range_ * 1.1
    # Support levels
    s3 = prev_close - range_ * 1.1 / 2
    s4 = prev_close - range_ * 1.1
    
    # Weekly EMA for trend filter (21-period)
    weekly_close = df_1w['close'].values
    ema_21 = pd.Series(weekly_close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align all to daily timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21)
    
    # Volume confirmation: volume > 1.3x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_21_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: close above R4, above weekly EMA, with volume
        if (close[i] > r4_aligned[i] and 
            close[i] > ema_21_aligned[i] and 
            vol_confirm[i] and 
            position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: close below S4, below weekly EMA, with volume
        elif (close[i] < s4_aligned[i] and 
              close[i] < ema_21_aligned[i] and 
              vol_confirm[i] and 
              position != -1):
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